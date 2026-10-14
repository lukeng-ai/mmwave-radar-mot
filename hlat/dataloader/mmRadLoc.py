import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Literal, Any
import os
from pathlib import Path
from abc import abstractmethod
import csv
from itertools import permutations

class mmRadLoc(Dataset):

    def __init__(self, mode: str, datasetPath: str, slicesDataPath: str, config_dict, transform=None, seq_length=None, use_pc_components=None, augmentation=None, data_part=1, use_ram=True, config=None):
        """
        Creates the dataset
        """
        super().__init__()
        print('Loading tracking dataset')
        self.transform = transform
        self.mode = mode
        self.datasetPath = datasetPath
        self.slicesDataPath = slicesDataPath
        self.config_dict = config_dict
        self.seq_length = seq_length
        self.use_pc_components = use_pc_components
        self.data_part = data_part
        self.use_ram = use_ram
        self.config = config
        self.return_sequence_targets = bool(getattr(self.config, 'RETURN_SEQUENCE_TARGETS', False) or any((tag in str(getattr(self.config, 'MODEL', '')).upper() for tag in ('PNTKPCONVTBA',))))
        self.stabilize_gt_tracks = bool(getattr(self.config, 'GT_TRACK_STABILIZATION_FLAG', False))
        self.gt_track_stabilization_threshold = float(getattr(self.config, 'GT_TRACK_STABILIZATION_THRESH', 0.2))
        self.gt_track_stabilization_max_missing = int(getattr(self.config, 'GT_TRACK_STABILIZATION_MAX_MISSING', 2))
        self.temporal_frame_sampling = bool(self.mode == 'train' and getattr(self.config, 'TEMPORAL_FRAME_SAMPLING_FLAG', False))
        self.temporal_frame_sampling_min_interval = int(getattr(self.config, 'TEMPORAL_FRAME_SAMPLING_MIN_INTERVAL', 1))
        self.temporal_frame_sampling_max_interval = int(getattr(self.config, 'TEMPORAL_FRAME_SAMPLING_MAX_INTERVAL', 10))
        self.input_point_drop_percent = float(getattr(self.config, 'PNT_INPUT_POINT_DROP_PERCENT', 0.0) or 0.0)
        assert seq_length is None or seq_length > 0
        if not 0.0 <= self.input_point_drop_percent <= 100.0:
            raise ValueError('PNT_INPUT_POINT_DROP_PERCENT must be between 0 and 100.')
        if self.temporal_frame_sampling_min_interval < 1:
            raise ValueError('TEMPORAL_FRAME_SAMPLING_MIN_INTERVAL must be at least 1.')
        if self.temporal_frame_sampling_max_interval < self.temporal_frame_sampling_min_interval:
            raise ValueError('TEMPORAL_FRAME_SAMPLING_MAX_INTERVAL must be greater than or equal to TEMPORAL_FRAME_SAMPLING_MIN_INTERVAL.')
        if self.temporal_frame_sampling and augmentation:
            raise ValueError('Temporal frame sampling cannot currently be combined with the legacy static mmRadLoc augmentations.')
        self.num_features = self.config_dict['num_features']
        self.num_people = self.config_dict.get('num_people', 2)
        self.features = []
        self.targets = []
        self.recording_sequence_lengths = []
        self.temporal_recordings = []
        self.temporal_sample_indices = []
        self.num_normal_temporal_samples = 0
        self.num_augmented_temporal_samples = 0
        if self.mode == 'train':
            self.recordings_list = config_dict['train_records']
        if self.mode == 'val':
            self.recordings_list = config_dict['val_records']
        if self.mode == 'test':
            self.recordings_list = config_dict['test_records']
        if not self.use_ram:
            cache_root = Path(self.slicesDataPath).expanduser().resolve()
            protected_paths = {
                Path(cache_root.anchor),
                Path.home().resolve(),
                Path.cwd().resolve(),
                Path(self.datasetPath).expanduser().resolve(),
            }
            if cache_root in protected_paths:
                raise ValueError('DATA_SLICE_PATH must point to a dedicated cache directory.')
            mode_cache_path = cache_root / self.mode
            radar_cache_path = mode_cache_path / 'radar'
            target_cache_path = mode_cache_path / 'ground_truth'
            radar_cache_path.mkdir(parents=True, exist_ok=True)
            target_cache_path.mkdir(parents=True, exist_ok=True)
            for cache_path in (radar_cache_path, target_cache_path):
                for cached_slice in cache_path.glob('*.npy'):
                    cached_slice.unlink()
        counter = 0
        for recording in self.recordings_list:
            pointcloud_frames = np.load(os.path.join(self.datasetPath + recording, getattr(self.config, 'DATA_POINT_FILENAME', 'radar_pointcloud.npy'))).astype(np.float32)[:, :, :self.num_features]
            target_frames = np.load(os.path.join(self.datasetPath + recording, getattr(self.config, 'DATA_TARGET_FILENAME', 'labels_final_kalman.npy'))).astype(np.float32)
            if self.stabilize_gt_tracks:
                target_frames = self._stabilize_gt_tracks_for_recording(target_frames)
            n_frames = target_frames.shape[0]
            if self.temporal_frame_sampling:
                n_normal_sequences = max(0, n_frames - self.seq_length + 1)
                max_sequence_span = (self.seq_length - 1) * self.temporal_frame_sampling_max_interval
                n_augmented_sequences = max(0, n_frames - max_sequence_span)
                normal_limit = int(np.floor(n_normal_sequences * self.data_part))
                augmented_limit = int(np.floor(n_augmented_sequences * self.data_part))
                recording_index = len(self.temporal_recordings)
                self.temporal_recordings.append((pointcloud_frames.astype(np.float32), target_frames.astype(np.float32)))
                self.temporal_sample_indices.extend(((recording_index, start_index, False) for start_index in range(normal_limit)))
                self.temporal_sample_indices.extend(((recording_index, start_index, True) for start_index in range(augmented_limit)))
                self.num_normal_temporal_samples += normal_limit
                self.num_augmented_temporal_samples += augmented_limit
                self.recording_sequence_lengths.append(normal_limit + augmented_limit)
                continue
            n_sequences = n_frames - self.seq_length + 1
            seq_pointclouds = np.zeros(shape=(n_sequences, self.seq_length) + pointcloud_frames.shape[1:])
            seq_targets = np.zeros(shape=(n_sequences, self.seq_length, target_frames.shape[1]), dtype=np.float32)
            for i in range(n_sequences):
                seq_pointclouds[i] = pointcloud_frames[i:i + self.seq_length]
                seq_targets[i] = target_frames[i:i + self.seq_length]
            limit = np.floor(seq_pointclouds.shape[0] * self.data_part).astype(int)
            seq_pointclouds = seq_pointclouds[:limit, :]
            seq_targets = seq_targets[:limit, :]
            self.recording_sequence_lengths.append(int(limit))
            final_targets = seq_targets[:, -1, :]
            targets_to_store = seq_targets if self.return_sequence_targets else final_targets
            if self.use_ram:
                self.features.append(seq_pointclouds.astype(np.float32))
                self.targets.append(targets_to_store.astype(np.float32))
                if augmentation:
                    np.random.seed(42)
                    features = seq_pointclouds.astype(np.float32).copy()
                    targets = targets_to_store.astype(np.float32).copy()
                    if self.config.AUGMENTATION_NOISE:
                        noisy_features = features.copy()
                        noisy_features[:, :, :, 0:3] = noisy_features[:, :, :, 0:3] + self.config.AUGMENTATION_NOISE * np.random.randn(*features.shape[:-1] + (3,))
                        noisy_features[:, :, :, 3:7] = noisy_features[:, :, :, 3:7] + self.config.AUGMENTATION_NOISE * noisy_features[:, :, :, 3:7] * np.random.randn(*features.shape[:-1] + (4,))
                        self.features.append(noisy_features.astype(np.float32))
                        self.targets.append(targets.astype(np.float32))
                    if self.config.AUGMENTATION_FLIP:
                        flipped_features = features.copy()
                        flipped_features[:, :, :, self.config.AUGMENTATION_FLIP - 1] = -1 * features[:, :, :, self.config.AUGMENTATION_FLIP - 1]
                        flipped_targets = targets.copy()
                        objects = flipped_targets[..., 0:30].reshape(-1, 5, 6)
                        objects[:, :, self.config.AUGMENTATION_FLIP - 1] *= -1
                        objects[:, :, self.config.AUGMENTATION_FLIP - 1 + 3] *= -1
                        flipped_targets[..., 0:30] = objects.reshape(*flipped_targets.shape[:-1], 30)
                        self.features.append(flipped_features.astype(np.float32))
                        self.targets.append(flipped_targets.astype(np.float32))
                    if self.config.AUGMENTATION_TRANSLATE:
                        translate_features = features.copy()
                        translate_features[:, :, :, 1] += self.config.AUGMENTATION_TRANSLATE
                        translate_targets = targets.copy()
                        objects1 = translate_targets[..., 0:30].reshape(-1, 5, 6)
                        objects1[:, :, 1] += self.config.AUGMENTATION_TRANSLATE
                        translate_targets[..., 0:30] = objects1.reshape(*translate_targets.shape[:-1], 30)
                        self.features.append(translate_features.astype(np.float32))
                        self.targets.append(translate_targets.astype(np.float32))
                    if self.config.AUGMENTATION_YTRANSLATE:
                        translate_features = features.copy()
                        translate_features[:, :, :, 1] += self.config.AUGMENTATION_YTRANSLATE
                        translate_targets = targets.copy()
                        objects1 = translate_targets[..., 0:30].reshape(-1, 5, 6)
                        objects1[:, :, 1] += self.config.AUGMENTATION_YTRANSLATE
                        translate_targets[..., 0:30] = objects1.reshape(*translate_targets.shape[:-1], 30)
                        self.features.append(translate_features.astype(np.float32))
                        self.targets.append(translate_targets.astype(np.float32))
            else:
                for i in range(seq_pointclouds.shape[0]):
                    np.save(os.path.join(self.slicesDataPath, self.mode, 'radar', str(counter)), seq_pointclouds[i].astype(np.float32))
                    np.save(os.path.join(self.slicesDataPath, self.mode, 'ground_truth', str(counter)), targets_to_store[i].astype(np.float32))
                    counter += 1
                if augmentation:
                    np.random.seed(42)
                    features = seq_pointclouds.astype(np.float32).copy()
                    targets = targets_to_store.astype(np.float32).copy()
                    if self.config.AUGMENTATION_NOISE:
                        noisy_features = features.copy()
                        noisy_features[:, :, :, 0:3] = noisy_features[:, :, :, 0:3] + self.config.AUGMENTATION_NOISE * np.random.randn(*features.shape[:-1] + (3,))
                        noisy_features[:, :, :, 3:7] = noisy_features[:, :, :, 3:7] + self.config.AUGMENTATION_NOISE * noisy_features[:, :, :, 3:7] * np.random.randn(*features.shape[:-1] + (4,))
                        for i in range(noisy_features.shape[0]):
                            np.save(os.path.join(self.slicesDataPath, self.mode, 'radar', str(counter)), noisy_features[i].astype(np.float32))
                            np.save(os.path.join(self.slicesDataPath, self.mode, 'ground_truth', str(counter)), targets[i].astype(np.float32))
                            counter += 1
                    if self.config.AUGMENTATION_FLIP:
                        flipped_features = features.copy()
                        flipped_features[:, :, :, self.config.AUGMENTATION_FLIP - 1] = -1 * features[:, :, :, self.config.AUGMENTATION_FLIP - 1]
                        flipped_targets = targets.copy()
                        objects = flipped_targets[..., 0:30].reshape(-1, 5, 6)
                        objects[:, :, self.config.AUGMENTATION_FLIP - 1] *= -1
                        objects[:, :, self.config.AUGMENTATION_FLIP - 1 + 3] *= -1
                        flipped_targets[..., 0:30] = objects.reshape(*flipped_targets.shape[:-1], 30)
                        for i in range(flipped_features.shape[0]):
                            np.save(os.path.join(self.slicesDataPath, self.mode, 'radar', str(counter)), flipped_features[i].astype(np.float32))
                            np.save(os.path.join(self.slicesDataPath, self.mode, 'ground_truth', str(counter)), flipped_targets[i].astype(np.float32))
                            counter += 1
                    if self.config.AUGMENTATION_TRANSLATE:
                        translate_features = features.copy()
                        translate_features[:, :, :, 1] += self.config.AUGMENTATION_TRANSLATE
                        translate_targets = targets.copy()
                        objects1 = translate_targets[..., 0:30].reshape(-1, 5, 6)
                        objects1[:, :, 1] += self.config.AUGMENTATION_TRANSLATE
                        translate_targets[..., 0:30] = objects1.reshape(*translate_targets.shape[:-1], 30)
                        for i in range(translate_features.shape[0]):
                            np.save(os.path.join(self.slicesDataPath, self.mode, 'radar', str(counter)), translate_features[i].astype(np.float32))
                            np.save(os.path.join(self.slicesDataPath, self.mode, 'ground_truth', str(counter)), translate_targets[i].astype(np.float32))
                            counter += 1
                    if self.config.AUGMENTATION_YTRANSLATE:
                        translate_features = features.copy()
                        translate_features[:, :, :, 1] += self.config.AUGMENTATION_YTRANSLATE
                        translate_targets = targets.copy()
                        objects1 = translate_targets[..., 0:30].reshape(-1, 5, 6)
                        objects1[:, :, 1] += self.config.AUGMENTATION_YTRANSLATE
                        translate_targets[..., 0:30] = objects1.reshape(*translate_targets.shape[:-1], 30)
                        for i in range(translate_features.shape[0]):
                            np.save(os.path.join(self.slicesDataPath, self.mode, 'radar', str(counter)), translate_features[i].astype(np.float32))
                            np.save(os.path.join(self.slicesDataPath, self.mode, 'ground_truth', str(counter)), translate_targets[i].astype(np.float32))
                            counter += 1
        if self.temporal_frame_sampling:
            print('Data loaded with dynamic temporal frame sampling:')
            print('recordings:', len(self.temporal_recordings), 'normal sequences:', self.num_normal_temporal_samples, 'additional augmented sequences:', self.num_augmented_temporal_samples, 'interval:', f'[{self.temporal_frame_sampling_min_interval}, {self.temporal_frame_sampling_max_interval}]')
        elif self.use_ram:

            def feature_generator():
                for feature in self.features:
                    yield feature.astype(np.float32)

            def targets_generator():
                for target in self.targets:
                    yield target.astype(np.float32)
            min_dim = min((target.shape[-1] for target in self.targets))
            trimmed_targets = [target[..., :min_dim] for target in self.targets]
            self.features = np.concatenate(list(feature_generator())).astype(np.float32)
            self.targets = np.concatenate(trimmed_targets).astype(np.float32)
            print('Data loaded:')
            print('recordings: len(self.features):', len(self.features), 'len(self.targets):', len(self.targets))
            assert len(self.features) == len(self.targets)
        else:
            print(f'{self.mode} data with {counter} files generated!')
            self.counter = counter
        if self.num_people is not None:
            self.num_people = self.num_people
        else:
            self.num_people = 5

    def __str__(self) -> str:
        return super().__str__()

    @staticmethod
    def _linear_sum_assignment_small(cost_matrix):
        if cost_matrix.size == 0:
            return ([], [])
        (num_rows, num_cols) = cost_matrix.shape
        if num_rows <= num_cols:
            best_cols = None
            best_cost = None
            for cols in permutations(range(num_cols), num_rows):
                cost = float(sum((cost_matrix[row, col] for (row, col) in enumerate(cols))))
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_cols = cols
            return (list(range(num_rows)), list(best_cols))
        (cols, rows) = mmRadLoc._linear_sum_assignment_small(cost_matrix.T)
        return (rows, cols)

    def _stabilize_gt_tracks_for_recording(self, target_frames):
        if target_frames.ndim != 2 or target_frames.shape[1] < 35:
            return target_frames
        stabilized = target_frames.copy()
        raw_objects = target_frames[:, :30].reshape(target_frames.shape[0], 5, 6)
        raw_classification = target_frames[:, 30:35]
        num_slots = min(5, int(getattr(self.config, 'GT_TRACK_STABILIZATION_NUM_SLOTS', self.num_people)))
        stable_objects = raw_objects.copy()
        stable_classification = raw_classification.copy()
        stable_objects[:, :num_slots] = 0.0
        stable_classification[:, :num_slots] = 0.0
        previous_positions = {}
        previous_missing = {}
        threshold = self.gt_track_stabilization_threshold
        max_missing = self.gt_track_stabilization_max_missing
        for frame_idx in range(target_frames.shape[0]):
            current_valid = (raw_classification[frame_idx, :num_slots] >= 0.5) & (np.abs(raw_objects[frame_idx, :num_slots, :2]).sum(axis=-1) > 1e-06)
            raw_indices = np.nonzero(current_valid)[0].tolist()
            assigned_raw = set()
            current_positions = {}
            if previous_positions and raw_indices:
                previous_slots = sorted(previous_positions.keys())
                previous_xy = np.stack([previous_positions[slot] for slot in previous_slots], axis=0)
                current_xy = raw_objects[frame_idx, raw_indices, :2]
                cost = np.linalg.norm(previous_xy[:, None, :] - current_xy[None, :, :], axis=-1)
                (row_ind, col_ind) = self._linear_sum_assignment_small(cost)
                for (row, col) in zip(row_ind, col_ind):
                    distance = float(cost[row, col])
                    if distance > threshold:
                        continue
                    stable_slot = int(previous_slots[row])
                    raw_slot = int(raw_indices[col])
                    stable_objects[frame_idx, stable_slot] = raw_objects[frame_idx, raw_slot]
                    stable_classification[frame_idx, stable_slot] = raw_classification[frame_idx, raw_slot]
                    current_positions[stable_slot] = raw_objects[frame_idx, raw_slot, :2].copy()
                    assigned_raw.add(raw_slot)
            free_slots = [slot for slot in range(num_slots) if slot not in current_positions]
            for raw_slot in raw_indices:
                if raw_slot in assigned_raw or not free_slots:
                    continue
                stable_slot = free_slots.pop(0)
                stable_objects[frame_idx, stable_slot] = raw_objects[frame_idx, raw_slot]
                stable_classification[frame_idx, stable_slot] = raw_classification[frame_idx, raw_slot]
                current_positions[stable_slot] = raw_objects[frame_idx, raw_slot, :2].copy()
            if max_missing > 0:
                retained_positions = {}
                retained_missing = {}
                for (stable_slot, position) in previous_positions.items():
                    if stable_slot in current_positions:
                        retained_positions[stable_slot] = current_positions[stable_slot]
                        retained_missing[stable_slot] = 0
                        continue
                    missing_count = previous_missing.get(stable_slot, 0) + 1
                    if missing_count <= max_missing:
                        retained_positions[stable_slot] = position
                        retained_missing[stable_slot] = missing_count
                for (stable_slot, position) in current_positions.items():
                    retained_positions[stable_slot] = position
                    retained_missing[stable_slot] = 0
                previous_positions = retained_positions
                previous_missing = retained_missing
            else:
                previous_positions = current_positions
                previous_missing = {stable_slot: 0 for stable_slot in current_positions}
        stabilized[:, :30] = stable_objects.reshape(target_frames.shape[0], 30)
        stabilized[:, 30:35] = stable_classification
        return stabilized

    def _drop_input_points(self, radar):
        if self.input_point_drop_percent <= 0.0:
            return radar
        degraded = radar.copy()
        for frame in degraded:
            valid_indices = np.flatnonzero(np.any(frame != 0.0, axis=-1))
            if valid_indices.size == 0:
                continue
            num_drop = int(np.floor(valid_indices.size * self.input_point_drop_percent / 100.0 + 0.5))
            num_drop = min(num_drop, valid_indices.size)
            if num_drop == 0:
                continue
            drop_indices = np.random.choice(valid_indices, size=num_drop, replace=False)
            frame[drop_indices] = 0.0
        return degraded

    def __len__(self) -> int:
        if self.temporal_frame_sampling:
            return len(self.temporal_sample_indices)
        elif self.use_ram:
            return self.targets.shape[0]
        else:
            return self.counter

    def __getitem__(self, index: int) -> Any:
        if self.temporal_frame_sampling:
            (recording_index, start_index, augmented) = self.temporal_sample_indices[index]
            (pointcloud_frames, target_frames) = self.temporal_recordings[recording_index]
            frame_interval = 1
            if augmented:
                frame_interval = np.random.randint(self.temporal_frame_sampling_min_interval, self.temporal_frame_sampling_max_interval + 1)
            frame_indices = start_index + np.arange(self.seq_length) * frame_interval
            radar = pointcloud_frames[frame_indices]
            sequence_targets = target_frames[frame_indices]
            ground_truth = sequence_targets if self.return_sequence_targets else sequence_targets[-1]
            radar = self._drop_input_points(radar)
            return (radar.astype(np.float32), ground_truth.astype(np.float32))
        elif self.use_ram:
            radar = self._drop_input_points(self.features[index])
            return (radar, self.targets[index])
        else:
            radar = np.load(os.path.join(self.slicesDataPath, self.mode, 'radar', str(index) + '.npy'))
            ground_truth = np.load(os.path.join(self.slicesDataPath, self.mode, 'ground_truth', str(index) + '.npy'))
            radar = self._drop_input_points(radar)
            return (radar, ground_truth)
