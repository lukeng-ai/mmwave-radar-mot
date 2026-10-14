import json
import os
import time
from collections import Counter, defaultdict
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def _to_tensor(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value)

def _to_bool_tensor(value):
    return _to_tensor(value).bool()

def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for (key, val) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if hasattr(value, 'item'):
        return value.item()
    return value

def tracking_validation_score(metrics, selection_metric='score', id_switch_weight=0.2, mpe_xy_weight=0.1, mpe_xy_normalization=None):
    selection_metric = str(selection_metric).strip().lower()
    if selection_metric == 'score':
        hota = float(metrics.get('hota') or 0.0)
        normalized_mean_position_error_xy = normalize_mean_position_error_xy(metrics, normalization=mpe_xy_normalization)
        return hota - float(mpe_xy_weight) * normalized_mean_position_error_xy
    metric_aliases = {'ass_a': 'assa', 'association_accuracy': 'assa'}
    metric_key = metric_aliases.get(selection_metric, selection_metric)
    supported_metrics = {'f1', 'mota', 'hota', 'assa', 'precision', 'recall'}
    if metric_key not in supported_metrics:
        supported = ', '.join(['score', *sorted(supported_metrics)])
        raise ValueError(f'Unsupported tracking validation selection metric {selection_metric!r}. Choose one of: {supported}.')
    value = metrics.get(metric_key)
    if value is None:
        raise ValueError(f'Tracking validation metric {metric_key!r} is unavailable for model selection.')
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f'Tracking validation metric {metric_key!r} must be finite, got {value}.')
    return value

def normalize_mean_position_error_xy(metrics, normalization=None, missing_value=1.0):
    if normalization is None:
        normalization = metrics.get('match_threshold', 1.0)
    normalization = float(normalization)
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError(f'MPE xy normalization must be finite and greater than zero, got {normalization}.')
    mean_position_error_xy = metrics.get('mean_position_error_xy')
    if mean_position_error_xy is None:
        return float(missing_value)
    mean_position_error_xy = float(mean_position_error_xy)
    if not np.isfinite(mean_position_error_xy) or mean_position_error_xy < 0:
        raise ValueError(f'Mean position error xy must be finite and non-negative, got {mean_position_error_xy}.')
    return mean_position_error_xy / normalization

def infer_track_ids_from_active_segments(active_mask):
    active_mask = _to_bool_tensor(active_mask)
    if active_mask.ndim != 2:
        raise ValueError(f'Expected active_mask [F, P], got {tuple(active_mask.shape)}')
    track_ids = torch.full(active_mask.shape, -1, dtype=torch.long)
    next_track_id = 0
    for slot_idx in range(active_mask.shape[1]):
        active_segment = False
        current_track_id = -1
        for frame_idx in range(active_mask.shape[0]):
            if bool(active_mask[frame_idx, slot_idx].item()):
                if not active_segment:
                    current_track_id = next_track_id
                    next_track_id += 1
                    active_segment = True
                track_ids[frame_idx, slot_idx] = current_track_id
            else:
                active_segment = False
    return track_ids.numpy()

def infer_gt_track_ids_from_positions(gt_regression, gt_valid, match_dims=2, gt_stabilization_threshold=0.2):
    gt_regression = _to_tensor(gt_regression).float()
    gt_valid = _to_bool_tensor(gt_valid)
    if gt_regression.ndim != 3:
        raise ValueError(f'Expected gt_regression [F, G, D], got {tuple(gt_regression.shape)}')
    if gt_valid.ndim != 2:
        raise ValueError(f'Expected gt_valid [F, G], got {tuple(gt_valid.shape)}')
    track_ids = torch.full(gt_valid.shape, -1, dtype=torch.long)
    previous_gt_tracks = {}
    next_gt_track_id = 0
    dims = min(int(match_dims), gt_regression.shape[-1])
    threshold = float(gt_stabilization_threshold)
    for frame_idx in range(gt_regression.shape[0]):
        gt_indices = torch.nonzero(gt_valid[frame_idx], as_tuple=False).flatten()
        gt_slots = [int(gt_idx.item()) for gt_idx in gt_indices]
        if not gt_slots:
            previous_gt_tracks = {}
            continue
        current_positions = {gt_slot: gt_regression[frame_idx, gt_slot, :dims].detach().cpu().numpy() for gt_slot in gt_slots}
        assigned_track_ids = {}
        assigned_current_slots = set()
        previous_items = list(previous_gt_tracks.items())
        if previous_items:
            previous_ids = [track_id for (track_id, _) in previous_items]
            previous_positions = np.stack([position for (_, position) in previous_items], axis=0)
            current_slot_list = list(current_positions.keys())
            current_position_array = np.stack([current_positions[slot] for slot in current_slot_list], axis=0)
            dist = np.linalg.norm(previous_positions[:, None, :] - current_position_array[None, :, :], axis=-1)
            (row_ind, col_ind) = linear_sum_assignment(dist)
            for (row, col) in zip(row_ind, col_ind):
                distance = float(dist[row, col])
                if distance > threshold:
                    continue
                track_id = int(previous_ids[row])
                gt_slot = int(current_slot_list[col])
                assigned_track_ids[gt_slot] = track_id
                assigned_current_slots.add(gt_slot)
        for gt_slot in gt_slots:
            if gt_slot in assigned_current_slots:
                continue
            assigned_track_ids[gt_slot] = next_gt_track_id
            next_gt_track_id += 1
        for (gt_slot, track_id) in assigned_track_ids.items():
            track_ids[frame_idx, gt_slot] = track_id
        previous_gt_tracks = {assigned_track_ids[gt_slot]: current_positions[gt_slot] for gt_slot in gt_slots}
    return track_ids.numpy()

class TrackingMetricsAccumulator:

    def __init__(self, score_threshold=0.5, match_threshold=1.0, match_dims=2, scores_are_logits=False, stabilize_gt=False, gt_stabilization_threshold=None):
        self.score_threshold = float(score_threshold)
        self.match_threshold = float(match_threshold)
        self.match_dims = int(match_dims)
        self.scores_are_logits = bool(scores_are_logits)
        self.stabilize_gt = bool(stabilize_gt)
        self.gt_stabilization_threshold = 0.2 if gt_stabilization_threshold is None else float(gt_stabilization_threshold)
        self.stats = defaultdict(float)
        self.previous_assignment = {}
        self.previous_seen = set()
        self.previous_gt_tracks = {}
        self.next_gt_track_id = 0
        self.sequence_id = 0
        self.gt_track_counts = Counter()
        self.pred_track_counts = Counter()
        self.pair_counts = Counter()
        self.total_frames = 0
        self.start_time = time.time()

    def reset_sequence(self):
        self.sequence_id += 1
        self.previous_assignment = {}
        self.previous_seen = set()
        self.previous_gt_tracks = {}
        self.next_gt_track_id = 0

    def _scores_to_active(self, pred_scores):
        pred_scores = _to_tensor(pred_scores).float()
        if self.scores_are_logits:
            pred_scores = torch.sigmoid(pred_scores)
        return (pred_scores, pred_scores >= self.score_threshold)

    def _track_id_for_pred(self, pred_idx, pred_track_ids):
        if pred_track_ids is None:
            return int(pred_idx)
        if torch.is_tensor(pred_track_ids):
            return int(pred_track_ids[pred_idx].item())
        return int(pred_track_ids[pred_idx])

    def _assign_gt_track_ids(self, gt_pos, gt_indices, gt_track_ids=None):
        gt_slots = [int(gt_idx.item()) for gt_idx in gt_indices]
        if gt_track_ids is not None:
            gt_track_ids = _to_tensor(gt_track_ids).long()
            return {gt_slot: int(gt_track_ids[gt_slot].item()) for gt_slot in gt_slots if int(gt_track_ids[gt_slot].item()) >= 0}
        if not self.stabilize_gt:
            return {gt_slot: gt_slot for gt_slot in gt_slots}
        if not gt_slots:
            self.previous_gt_tracks = {}
            return {}
        dims = min(self.match_dims, gt_pos.shape[-1])
        current_positions = {gt_slot: gt_pos[gt_slot, :dims].detach().cpu().numpy() for gt_slot in gt_slots}
        assigned_track_ids = {}
        assigned_current_slots = set()
        previous_items = list(self.previous_gt_tracks.items())
        if previous_items:
            previous_ids = [track_id for (track_id, _) in previous_items]
            previous_positions = np.stack([position for (_, position) in previous_items], axis=0)
            current_slot_list = list(current_positions.keys())
            current_position_array = np.stack([current_positions[slot] for slot in current_slot_list], axis=0)
            dist = np.linalg.norm(previous_positions[:, None, :] - current_position_array[None, :, :], axis=-1)
            (row_ind, col_ind) = linear_sum_assignment(dist)
            for (row, col) in zip(row_ind, col_ind):
                distance = float(dist[row, col])
                if distance > self.gt_stabilization_threshold:
                    continue
                track_id = int(previous_ids[row])
                gt_slot = int(current_slot_list[col])
                assigned_track_ids[gt_slot] = track_id
                assigned_current_slots.add(gt_slot)
        for gt_slot in gt_slots:
            if gt_slot in assigned_current_slots:
                continue
            assigned_track_ids[gt_slot] = self.next_gt_track_id
            self.next_gt_track_id += 1
        self.previous_gt_tracks = {assigned_track_ids[gt_slot]: current_positions[gt_slot] for gt_slot in gt_slots}
        return assigned_track_ids

    def update(self, pred_pos, pred_scores, gt_pos, gt_valid, pred_track_ids=None, gt_track_ids=None):
        pred_pos = _to_tensor(pred_pos).float()
        gt_pos = _to_tensor(gt_pos).float()
        gt_valid = _to_bool_tensor(gt_valid)
        (pred_scores, pred_active) = self._scores_to_active(pred_scores)
        pred_indices = torch.nonzero(pred_active, as_tuple=False).flatten()
        gt_indices = torch.nonzero(gt_valid, as_tuple=False).flatten()
        frame_stats = defaultdict(float)
        frame_stats['num_predictions'] = int(pred_indices.numel())
        frame_stats['num_gt'] = int(gt_indices.numel())
        frame_stats['count_abs_error'] = abs(frame_stats['num_predictions'] - frame_stats['num_gt'])
        pred_active_track_ids = {int(pred_idx.item()): self._track_id_for_pred(int(pred_idx.item()), pred_track_ids) for pred_idx in pred_indices}
        gt_slot_to_track_id = self._assign_gt_track_ids(gt_pos, gt_indices, gt_track_ids=gt_track_ids)
        gt_active_ids = list(gt_slot_to_track_id.values())
        scoped_pred_track_ids = {pred_idx: (self.sequence_id, track_id) for (pred_idx, track_id) in pred_active_track_ids.items()}
        scoped_gt_track_ids = {gt_slot: (self.sequence_id, track_id) for (gt_slot, track_id) in gt_slot_to_track_id.items()}
        for track_id in scoped_pred_track_ids.values():
            self.pred_track_counts[track_id] += 1
        for track_id in scoped_gt_track_ids.values():
            self.gt_track_counts[track_id] += 1
        matches = []
        assigned_preds = set()
        assigned_gt = set()
        if pred_indices.numel() > 0 and gt_indices.numel() > 0:
            dims = min(self.match_dims, pred_pos.shape[-1], gt_pos.shape[-1])
            pred_match_pos = pred_pos[pred_indices, :dims]
            gt_match_pos = gt_pos[gt_indices, :dims]
            dist = torch.cdist(pred_match_pos, gt_match_pos, p=2).numpy()
            (row_ind, col_ind) = linear_sum_assignment(dist)
            for (row, col) in zip(row_ind, col_ind):
                distance = float(dist[row, col])
                if distance > self.match_threshold:
                    continue
                pred_idx = int(pred_indices[row].item())
                gt_idx = int(gt_indices[col].item())
                assigned_preds.add(pred_idx)
                assigned_gt.add(gt_idx)
                matches.append((pred_idx, gt_idx, distance))
                diff = pred_pos[pred_idx] - gt_pos[gt_idx]
                xyz_error = float(torch.linalg.norm(diff[:min(3, diff.shape[-1])]).item())
                xy_error = float(torch.linalg.norm(diff[:min(2, diff.shape[-1])]).item())
                frame_stats['sum_error_xyz'] += xyz_error
                frame_stats['sum_sq_error_xyz'] += xyz_error ** 2
                frame_stats['sum_error_xy'] += xy_error
                frame_stats['sum_sq_error_xy'] += xy_error ** 2
                frame_stats['matched_distance_count'] += 1
                pred_track_id = scoped_pred_track_ids[pred_idx]
                gt_track_id = scoped_gt_track_ids[gt_idx]
                self.pair_counts[gt_track_id, pred_track_id] += 1
            frame_stats['tp'] = len(matches)
            frame_stats['fp'] = int(pred_indices.numel()) - len(assigned_preds)
            frame_stats['fn'] = int(gt_indices.numel()) - len(assigned_gt)
        else:
            frame_stats['tp'] = 0
            frame_stats['fp'] = int(pred_indices.numel())
            frame_stats['fn'] = int(gt_indices.numel())
        current_assignment = {}
        for (pred_idx, gt_idx, _) in matches:
            current_assignment[gt_slot_to_track_id[gt_idx]] = pred_active_track_ids[pred_idx]
        current_seen = set(current_assignment.keys())
        for (gt_idx, pred_track_id) in current_assignment.items():
            if gt_idx in self.previous_assignment:
                frame_stats['id_continuations'] += 1
                if self.previous_assignment[gt_idx] != pred_track_id:
                    frame_stats['id_switches'] += 1
        for gt_idx in self.previous_seen:
            if gt_idx not in current_seen and gt_idx in gt_active_ids:
                frame_stats['track_fragmentations'] += 1
        self.previous_assignment = current_assignment
        self.previous_seen = current_seen
        for (key, value) in frame_stats.items():
            self.stats[key] += value
        self.total_frames += 1
        return (frame_stats, matches)

    def finalize(self, num_sequences=None, sequence_length=None, runtime_sec=None):
        runtime_sec = time.time() - self.start_time if runtime_sec is None else runtime_sec
        tp = self.stats['tp']
        fp = self.stats['fp']
        fn = self.stats['fn']
        num_gt = self.stats['num_gt']
        precision = tp / (tp + fp + 1e-09)
        recall = tp / (tp + fn + 1e-09)
        f1 = 2 * precision * recall / (precision + recall + 1e-09)
        matched_count = self.stats['matched_distance_count']
        mean_xyz = self.stats['sum_error_xyz'] / matched_count if matched_count else None
        rmse_xyz = np.sqrt(self.stats['sum_sq_error_xyz'] / matched_count) if matched_count else None
        mean_xy = self.stats['sum_error_xy'] / matched_count if matched_count else None
        rmse_xy = np.sqrt(self.stats['sum_sq_error_xy'] / matched_count) if matched_count else None
        id_switch_rate = self.stats['id_switches'] / self.stats['id_continuations'] if self.stats['id_continuations'] else None
        mota = 1.0 - (fn + fp + self.stats['id_switches']) / (num_gt + 1e-09) if num_gt else None
        detection_accuracy = tp / (tp + fp + fn + 1e-09)
        association_values = []
        for (gt_id, pred_track_id) in self.pair_counts:
            pair_count = self.pair_counts[gt_id, pred_track_id]
            union_count = self.gt_track_counts[gt_id] + self.pred_track_counts[pred_track_id] - pair_count
            ass_a = pair_count / (union_count + 1e-09)
            association_values.extend([ass_a] * pair_count)
        association_accuracy = float(np.mean(association_values)) if association_values else 0.0
        hota = float(np.sqrt(detection_accuracy * association_accuracy))
        position_metrics = {'mpe_xyz': None if mean_xyz is None else float(mean_xyz), 'mae_xyz': None if mean_xyz is None else float(mean_xyz), 'rmse_xyz': None if rmse_xyz is None else float(rmse_xyz), 'mpe_xy': None if mean_xy is None else float(mean_xy), 'mae_xy': None if mean_xy is None else float(mean_xy), 'rmse_xy': None if rmse_xy is None else float(rmse_xy)}
        classification_metrics = {'hota': float(hota), 'mota': None if mota is None else float(mota), 'accuracy': float(detection_accuracy), 'deta': float(detection_accuracy), 'assa': float(association_accuracy), 'AssA': float(association_accuracy), 'association_accuracy': float(association_accuracy), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}
        metrics = {'num_sequences': None if num_sequences is None else int(num_sequences), 'sequence_length': None if sequence_length is None else int(sequence_length), 'num_evaluated_frames': int(self.total_frames), 'score_threshold': float(self.score_threshold), 'match_threshold': float(self.match_threshold), 'match_dims': int(self.match_dims), 'stabilize_gt': bool(self.stabilize_gt), 'gt_stabilization_threshold': float(self.gt_stabilization_threshold), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'accuracy': float(detection_accuracy), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'num_gt': int(num_gt), 'mean_position_error_xyz': None if mean_xyz is None else float(mean_xyz), 'rmse_position_error_xyz': None if rmse_xyz is None else float(rmse_xyz), 'mean_position_error_xy': None if mean_xy is None else float(mean_xy), 'rmse_position_error_xy': None if rmse_xy is None else float(rmse_xy), 'mean_abs_count_error': float(self.stats['count_abs_error'] / max(self.total_frames, 1)), 'id_switches': int(self.stats['id_switches']), 'id_continuations': int(self.stats['id_continuations']), 'id_switch_rate': None if id_switch_rate is None else float(id_switch_rate), 'track_fragmentations': int(self.stats['track_fragmentations']), 'mota': None if mota is None else float(mota), 'hota': float(hota), 'detection_accuracy': float(detection_accuracy), 'deta': float(detection_accuracy), 'assa': float(association_accuracy), 'AssA': float(association_accuracy), 'association_accuracy': float(association_accuracy), 'classification_metrics': classification_metrics, 'position_metrics': position_metrics, 'mpe_xyz': position_metrics['mpe_xyz'], 'mae_xyz': position_metrics['mae_xyz'], 'rmse_xyz': position_metrics['rmse_xyz'], 'mpe_xy': position_metrics['mpe_xy'], 'mae_xy': position_metrics['mae_xy'], 'rmse_xy': position_metrics['rmse_xy'], 'runtime_sec': float(runtime_sec), 'fps': float(self.total_frames / runtime_sec) if runtime_sec > 0 else None}
        return metrics

def compute_tracking_metrics_from_arrays(pred_regression, pred_scores, gt_regression, gt_valid, score_threshold=0.5, match_threshold=1.0, match_dims=2, pred_track_ids=None, gt_track_ids=None, scores_are_logits=False, sequence_breaks=None, stabilize_gt=False, gt_stabilization_threshold=None, evaluation_mask=None):
    pred_regression = _to_tensor(pred_regression).float()
    pred_scores = _to_tensor(pred_scores).float()
    gt_regression = _to_tensor(gt_regression).float()
    gt_valid = _to_bool_tensor(gt_valid)
    if pred_regression.ndim != 3:
        raise ValueError(f'Expected pred_regression [F, P, D], got {tuple(pred_regression.shape)}')
    if gt_regression.ndim != 3:
        raise ValueError(f'Expected gt_regression [F, G, D], got {tuple(gt_regression.shape)}')
    if pred_scores.ndim != 2:
        raise ValueError(f'Expected pred_scores [F, P], got {tuple(pred_scores.shape)}')
    if gt_valid.ndim != 2:
        raise ValueError(f'Expected gt_valid [F, G], got {tuple(gt_valid.shape)}')
    if evaluation_mask is None:
        evaluation_mask = torch.ones(pred_regression.shape[0], dtype=torch.bool)
    else:
        evaluation_mask = _to_bool_tensor(evaluation_mask).flatten()
        if evaluation_mask.shape[0] != pred_regression.shape[0]:
            raise ValueError(f'evaluation_mask must contain one value per frame: {evaluation_mask.shape[0]} != {pred_regression.shape[0]}.')
    accumulator = TrackingMetricsAccumulator(score_threshold=score_threshold, match_threshold=match_threshold, match_dims=match_dims, scores_are_logits=scores_are_logits, stabilize_gt=stabilize_gt, gt_stabilization_threshold=gt_stabilization_threshold)
    sequence_breaks = set(sequence_breaks or [])
    pending_reset = False
    started_sequence = False
    num_sequences = 0
    for frame_idx in range(pred_regression.shape[0]):
        if frame_idx in sequence_breaks:
            pending_reset = True
        if not bool(evaluation_mask[frame_idx].item()):
            pending_reset = True
            continue
        if not started_sequence:
            started_sequence = True
            num_sequences = 1
            pending_reset = False
        elif pending_reset:
            accumulator.reset_sequence()
            num_sequences += 1
            pending_reset = False
        frame_track_ids = pred_track_ids
        if pred_track_ids is not None:
            frame_track_ids = pred_track_ids[frame_idx] if np.asarray(pred_track_ids).ndim == 2 else pred_track_ids
        frame_gt_track_ids = gt_track_ids
        if gt_track_ids is not None:
            frame_gt_track_ids = gt_track_ids[frame_idx] if np.asarray(gt_track_ids).ndim == 2 else gt_track_ids
        accumulator.update(pred_regression[frame_idx], pred_scores[frame_idx], gt_regression[frame_idx], gt_valid[frame_idx], pred_track_ids=frame_track_ids, gt_track_ids=frame_gt_track_ids)
    metrics = accumulator.finalize(num_sequences=num_sequences, sequence_length=pred_regression.shape[0])
    metrics['num_ignored_frames'] = int((~evaluation_mask).sum().item())
    return metrics

def save_tracking_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(_json_safe(metrics), file, indent=2)
