import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import json
from scipy.optimize import linear_sum_assignment
from matplotlib.colors import hsv_to_rgb, to_rgb
from matplotlib.patches import Ellipse
if __package__ is None or __package__ == '':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
from hlat.utils.tracking_metrics import compute_tracking_metrics_from_arrays, infer_gt_track_ids_from_positions, infer_track_ids_from_active_segments, save_tracking_metrics
from hlat.utils.limit_prediction_for_testset import apply_y_limit, filter_sequences_and_equalize
BIG = 1000000000.0
KALMAN_TRACK_COLORS = ('#000000', '#008000', '#FFD700', '#696969', '#00A86B', '#808000', '#2E8B57', '#B8860B', '#7CFC00', '#A9A9A9', '#3CB371', '#556B2F')

def distinct_prediction_colors(count):
    """Return distinct Kalman colors without red or blue hues."""
    if count <= 0:
        return []
    colors = [to_rgb(color) for color in KALMAN_TRACK_COLORS[:count]]
    extra_count = count - len(colors)
    if extra_count <= 0:
        return np.asarray(colors)
    indices = np.arange(extra_count, dtype=float)
    hues = 0.15 + 0.23 * (indices * 0.61803398875 % 1.0)
    saturations = 0.45 + 0.5 * ((indices * 0.41421356237 + 0.2) % 1.0)
    values = 0.45 + 0.5 * ((indices * 0.73205080757 + 0.1) % 1.0)
    extra_colors = hsv_to_rgb(np.column_stack((hues, saturations, values)))
    return np.vstack((colors, extra_colors))

class KalmanFilter:

    def __init__(self, dt, process_noise_var, measurement_noise_var, dim, track_id=None):
        self.dt = dt
        self.n = 6
        if dim == 2:
            self.meas_idx = [0, 1]
        elif dim == 3:
            self.meas_idx = [0, 1, 2]
        else:
            raise ValueError('dim must be 2 or 3')
        self.m = len(self.meas_idx)
        self.F = torch.tensor([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=torch.float32)
        self.H = torch.zeros((self.m, self.n), dtype=torch.float32)
        for (r, c) in enumerate(self.meas_idx):
            self.H[r, c] = 1.0
        self.Q = float(process_noise_var) * torch.diag(torch.tensor([1, 1, 1, 80, 80, 80], dtype=torch.float32))
        self.R = float(measurement_noise_var) * torch.eye(self.m)
        self.x = torch.zeros(self.n, 1)
        self.P = torch.eye(self.n)
        self.die_counter = 0
        self.history = []
        self.start_frame = 0
        self.end_frame = -1
        self.is_new = True
        self.slot = None
        self.track_id = track_id

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.t() + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.t() + self.R
        K = self.P @ self.H.t() @ torch.inverse(S)
        self.x = self.x + K @ y
        I = torch.eye(self.n, dtype=torch.float32)
        self.P = (I - K @ self.H) @ self.P

def hungarian_assignment(cost_matrix: torch.Tensor, threshold: float):
    cost = cost_matrix.detach().cpu().numpy()
    (row_idx, col_idx) = linear_sum_assignment(cost)
    (num_filters, _) = cost_matrix.shape
    assignments = [None] * num_filters
    used_meas = set()
    for (i, j) in zip(row_idx, col_idx):
        if cost[i, j] < float(threshold):
            assignments[i] = int(j)
            used_meas.add(int(j))
    return (assignments, used_meas)

def apply_kalman(pred_pos_and_velo, pred_cls, var_process_noise, var_measurement_noise, threshold, max_miss, max_active_filters, filter_out_short_frames, dt=1 / 15, dim=2, pred_score_threshold=0.5, return_track_ids=False):
    if dim == 2:
        meas_idx = [0, 1]
        exist_idx = [0, 1]
    else:
        meas_idx = [0, 1, 2]
        exist_idx = [0, 1, 2]
    meas_dim = len(meas_idx)
    pred_pos_and_velo = torch.as_tensor(pred_pos_and_velo, dtype=torch.float32)
    if pred_pos_and_velo.ndim != 3 or pred_pos_and_velo.shape[-1] < 3:
        raise ValueError('pred_pos_and_velo must be (frames, people, 6>=).')
    pred_cls = torch.as_tensor(pred_cls, dtype=torch.float32)
    pred_active = pred_cls >= float(pred_score_threshold)
    exists_mask = torch.abs(pred_pos_and_velo[..., exist_idx]).sum(dim=-1) > 0
    use_mask = pred_active & exists_mask
    process_noise_var = float(var_process_noise)
    measurement_noise_var = float(var_measurement_noise)
    threshold = float(threshold)
    max_miss = int(max_miss)
    max_active_filters = int(max_active_filters)
    filter_out_short_frames = int(filter_out_short_frames)
    F = pred_pos_and_velo.shape[0]
    state_history = torch.zeros((F, max_active_filters, 6), dtype=torch.float32)
    filter_active = torch.zeros((F, max_active_filters), dtype=torch.int32)
    track_id_history = torch.full((F, max_active_filters), -1, dtype=torch.long)
    free_slots = list(range(max_active_filters))
    next_track_id = 0
    active_filters = []
    dead_filters_history = []
    all_filters = []
    mask0 = use_mask[0]
    init_idxs = torch.nonzero(mask0, as_tuple=False).flatten().tolist()
    for i in init_idxs:
        if not free_slots:
            break
        meas0 = pred_pos_and_velo[0, i, :3]
        if torch.allclose(meas0, torch.zeros_like(meas0)):
            continue
        slot = free_slots.pop(0)
        kf = KalmanFilter(dt, process_noise_var, measurement_noise_var, dim, track_id=next_track_id)
        next_track_id += 1
        kf.slot = slot
        kf.x = torch.cat([meas0, torch.zeros(3, dtype=torch.float32)], dim=0).view(6, 1)
        kf.P = torch.diag(torch.tensor([5, 5, 5, 25, 25, 25], dtype=torch.float32))
        kf.start_frame = 0
        kf.end_frame = 0
        kf.history.append(kf.x.flatten())
        active_filters.append(kf)
        state_history[0, slot, :] = kf.x.flatten()
        filter_active[0, slot] = 1
        track_id_history[0, slot] = kf.track_id
    for t in range(1, F):
        if active_filters:
            still_active = []
            for kf in active_filters:
                if kf.die_counter > max_miss:
                    dead_filters_history.append(kf.history)
                    all_filters.append(kf)
                    free_slots.append(kf.slot)
                else:
                    still_active.append(kf)
            active_filters = still_active
        for kf in active_filters:
            kf.predict()
        if meas_dim == 2:
            meas_idx = [0, 1]
        else:
            meas_idx = [0, 1, 2]
        meas_all = pred_pos_and_velo[t, :, :][:, meas_idx]
        avail = use_mask[t]
        meas_avail = meas_all[avail]
        nonzero = torch.abs(meas_avail).sum(dim=1) > 0
        meas_use = meas_avail[nonzero]
        M = meas_use.shape[0]
        if M == 0:
            for kf in active_filters:
                kf.die_counter += 1
                kf.history.append(kf.x.clone().flatten())
                kf.end_frame = t
                state_history[t, kf.slot, :] = kf.x.flatten()
                filter_active[t, kf.slot] = 1
                track_id_history[t, kf.slot] = kf.track_id
            still_active = []
            for kf in active_filters:
                if kf.die_counter >= max_miss:
                    dead_filters_history.append(kf.history)
                    all_filters.append(kf)
                    free_slots.append(kf.slot)
                else:
                    still_active.append(kf)
            active_filters = still_active
            continue
        if active_filters:
            num_filters = len(active_filters)
            cost_matrix = torch.full((num_filters, M), BIG, dtype=torch.float32)
            for (i, kf) in enumerate(active_filters):
                pred_pos_meas = kf.x[kf.meas_idx, 0]
                diffs = meas_use - pred_pos_meas.unsqueeze(0)
                cost_matrix[i] = torch.norm(diffs, dim=1)
            (assignments, used_meas) = hungarian_assignment(cost_matrix, threshold)
            used_meas = set(used_meas)
            for (i, kf) in enumerate(active_filters):
                if assignments[i] is not None:
                    z = meas_use[assignments[i]].view(meas_dim, 1)
                    kf.update(z)
                    kf.die_counter = 0
                    kf.is_new = False
                else:
                    kf.die_counter += 1
                kf.history.append(kf.x.clone().flatten())
                kf.end_frame = t
                state_history[t, kf.slot, :] = kf.x.flatten()
                filter_active[t, kf.slot] = 1
                track_id_history[t, kf.slot] = kf.track_id
        else:
            used_meas = set()
        still_active = []
        for kf in active_filters:
            if kf.die_counter >= max_miss:
                dead_filters_history.append(kf.history)
                all_filters.append(kf)
                free_slots.append(kf.slot)
            else:
                still_active.append(kf)
        active_filters = still_active
        for j in range(M):
            if j in used_meas:
                continue
            if not free_slots:
                break
            too_close = False
            for kf in active_filters:
                pred_pos = kf.x[kf.meas_idx, 0]
                d = torch.norm(meas_use[j] - pred_pos)
                if d < 0.5:
                    too_close = True
                    break
            if too_close:
                continue
            slot = free_slots.pop(0)
            kf = KalmanFilter(dt, process_noise_var, measurement_noise_var, dim, track_id=next_track_id)
            next_track_id += 1
            kf.slot = slot
            init_state = torch.zeros(6, 1, dtype=torch.float32)
            init_state[meas_idx, 0] = meas_use[j]
            kf.x = init_state
            kf.P = torch.diag(torch.tensor([5, 5, 5, 25, 25, 25], dtype=torch.float32))
            kf.start_frame = t
            kf.end_frame = t
            kf.is_new = True
            kf.history.append(kf.x.clone().flatten())
            active_filters.append(kf)
            state_history[t, slot, :] = kf.x.flatten()
            filter_active[t, slot] = 1
            track_id_history[t, slot] = kf.track_id
    for kf in active_filters:
        kf.end_frame = max(kf.end_frame, kf.start_frame)
        dead_filters_history.append(kf.history)
        all_filters.append(kf)
    if filter_out_short_frames:
        keep = []
        for kf in all_filters:
            if len(kf.history) <= filter_out_short_frames:
                filter_active[kf.start_frame:kf.end_frame + 1, kf.slot] = 0
                track_id_history[kf.start_frame:kf.end_frame + 1, kf.slot] = -1
            else:
                keep.append(kf)
        all_filters = keep
    full_kf_states = torch.zeros((F, max_active_filters, 6), dtype=torch.float32)
    for kf in all_filters:
        (s, e) = (kf.start_frame, max(kf.end_frame, kf.start_frame))
        hist = torch.stack(kf.history)
        full_kf_states[s:e + 1, kf.slot, :] = hist[:e - s + 1]
    result = (full_kf_states.numpy(), filter_active.numpy(), active_filters, dead_filters_history, all_filters)
    if return_track_ids:
        return (*result, track_id_history.numpy())
    return result

def _normalize_sequence_lengths(sequence_lengths, total_frames):
    if sequence_lengths is None:
        return [int(total_frames)]
    lengths = [int(length) for length in sequence_lengths if int(length) > 0]
    if sum(lengths) != int(total_frames):
        raise ValueError(f'Sequence lengths must sum to the number of prediction frames: {sum(lengths)} != {total_frames}.')
    return lengths

def apply_kalman_per_sequence(pred_pos_and_velo, pred_cls, var_process_noise, var_measurement_noise, threshold, max_miss, max_active_filters, filter_out_short_frames, sequence_lengths=None, dt=1 / 15, dim=2, pred_score_threshold=0.5):
    total_frames = int(pred_pos_and_velo.shape[0])
    lengths = _normalize_sequence_lengths(sequence_lengths, total_frames)
    regression_parts = []
    classification_parts = []
    track_id_parts = []
    all_active_filters = []
    all_dead_filter_histories = []
    all_filters = []
    next_track_id = 0
    start = 0
    for length in lengths:
        end = start + length
        (regression, classification, active, dead, filters, track_ids) = apply_kalman(pred_pos_and_velo[start:end], pred_cls[start:end], var_process_noise, var_measurement_noise, threshold, max_miss, max_active_filters, filter_out_short_frames, dt=dt, dim=dim, pred_score_threshold=pred_score_threshold, return_track_ids=True)
        valid_track_ids = track_ids >= 0
        if np.any(valid_track_ids):
            track_ids[valid_track_ids] += next_track_id
            next_track_id = int(track_ids[valid_track_ids].max()) + 1
        for kalman_filter in filters:
            kalman_filter.start_frame += start
            kalman_filter.end_frame += start
        regression_parts.append(regression)
        classification_parts.append(classification)
        track_id_parts.append(track_ids)
        all_active_filters.extend(active)
        all_dead_filter_histories.extend(dead)
        all_filters.extend(filters)
        start = end
    return (np.concatenate(regression_parts, axis=0), np.concatenate(classification_parts, axis=0), np.concatenate(track_id_parts, axis=0), all_active_filters, all_dead_filter_histories, all_filters, lengths)

def infer_track_ids_per_sequence(active_mask, sequence_lengths):
    track_id_parts = []
    next_track_id = 0
    start = 0
    for length in sequence_lengths:
        end = start + int(length)
        track_ids = infer_track_ids_from_active_segments(active_mask[start:end])
        valid_track_ids = track_ids >= 0
        if np.any(valid_track_ids):
            track_ids[valid_track_ids] += next_track_id
            next_track_id = int(track_ids[valid_track_ids].max()) + 1
        track_id_parts.append(track_ids)
        start = end
    return np.concatenate(track_id_parts, axis=0)

def infer_gt_track_ids_per_sequence(gt_regression, gt_valid, sequence_lengths, match_dims, stabilization_threshold):
    track_id_parts = []
    start = 0
    for length in sequence_lengths:
        end = start + int(length)
        track_id_parts.append(infer_gt_track_ids_from_positions(gt_regression[start:end], gt_valid[start:end], match_dims=match_dims, gt_stabilization_threshold=stabilization_threshold))
        start = end
    return np.concatenate(track_id_parts, axis=0)

def evaluate_kalman_tracking(pred_regression, pred_scores, gt_regression, gt_valid, sequence_lengths=None, *, var_process_noise=0.14, var_measurement_noise=3.0, assignment_threshold=0.4, max_miss=6, max_active_filters=2, filter_short_frames=0, kalman_dim=2, kalman_pred_score_threshold=0.5, tracking_score_threshold=0.5, tracking_match_threshold=1.0, tracking_match_dims=2, stabilize_gt=False, gt_stabilization_threshold=0.2, evaluation_y_limit=4.9, reset_between_sequences=True):
    """Run the stateful Kalman test evaluation used by the CLI.

    Kalman state advances over every frame. Frames outside ``evaluation_y_limit``
    are ignored only by the metric accumulator, matching
    ``--evaluation-y-limit-mode ignore``.
    """
    pred_regression = np.asarray(pred_regression)
    pred_scores = np.asarray(pred_scores)
    gt_regression = np.asarray(gt_regression)
    gt_valid = np.asarray(gt_valid, dtype=bool)
    lengths = _normalize_sequence_lengths(sequence_lengths, pred_regression.shape[0])
    kalman_lengths = lengths if reset_between_sequences else [pred_regression.shape[0]]
    (filtered_regression, filtered_scores, pred_track_ids, _, _, _, _) = apply_kalman_per_sequence(pred_regression, pred_scores, var_process_noise, var_measurement_noise, assignment_threshold, max_miss, max_active_filters, filter_short_frames, sequence_lengths=kalman_lengths, dt=1 / 15, dim=kalman_dim, pred_score_threshold=kalman_pred_score_threshold)
    gt_track_ids = None
    if stabilize_gt:
        gt_track_ids = infer_gt_track_ids_per_sequence(gt_regression, gt_valid, lengths, match_dims=tracking_match_dims, stabilization_threshold=gt_stabilization_threshold)
    evaluation_mask = np.ones(gt_regression.shape[0], dtype=bool)
    if evaluation_y_limit is not None:
        evaluation_mask = ~(gt_valid & (gt_regression[:, :, 1] > float(evaluation_y_limit))).any(axis=1)
    sequence_breaks = np.cumsum(lengths)[:-1].tolist() if reset_between_sequences else []
    metrics = compute_tracking_metrics_from_arrays(pred_regression=filtered_regression, pred_scores=filtered_scores, gt_regression=gt_regression, gt_valid=gt_valid, score_threshold=tracking_score_threshold, match_threshold=tracking_match_threshold, match_dims=tracking_match_dims, pred_track_ids=pred_track_ids, gt_track_ids=gt_track_ids, scores_are_logits=False, sequence_breaks=sequence_breaks, stabilize_gt=stabilize_gt and gt_track_ids is None, gt_stabilization_threshold=gt_stabilization_threshold, evaluation_mask=evaluation_mask)
    metrics.update({'evaluation_mode': 'kalman_stateful', 'pred_track_id_mode': 'kalman_filter_ids', 'kalman_var_process_noise': float(var_process_noise), 'kalman_var_measurement_noise': float(var_measurement_noise), 'kalman_assignment_threshold': float(assignment_threshold), 'kalman_pred_score_threshold': float(kalman_pred_score_threshold), 'kalman_max_miss': int(max_miss), 'kalman_max_active_filters': int(max_active_filters), 'kalman_dim': int(kalman_dim), 'recording_sequence_lengths': [int(length) for length in lengths], 'kalman_reset_between_sequences': bool(reset_between_sequences), 'tracking_reset_between_sequences': bool(reset_between_sequences), 'gt_track_id_mode': 'pre_filter_stabilized' if gt_track_ids is not None else 'slot_indices', 'evaluation_y_limit': None if evaluation_y_limit is None else float(evaluation_y_limit), 'evaluation_y_limit_mode': 'ignore', 'frames_over_evaluation_y_limit': int((~evaluation_mask).sum())})
    outputs = {'preds_regression': filtered_regression, 'preds_classification': filtered_scores, 'targets_regression': gt_regression, 'targets_classification': gt_valid.astype(np.float32), 'pred_track_ids': pred_track_ids, 'evaluation_valid_mask': evaluation_mask, 'recording_sequence_lengths': np.asarray(lengths, dtype=np.int64)}
    if gt_track_ids is not None:
        outputs['gt_track_ids'] = gt_track_ids
    return (metrics, outputs)

def save_kalman_tracking_result(result_path, metrics, outputs):
    os.makedirs(result_path, exist_ok=True)
    for (name, value) in outputs.items():
        np.save(os.path.join(result_path, f'{name}.npy'), value)
    save_tracking_metrics(metrics, os.path.join(result_path, 'tracking_metrics.json'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-vpn', '--var_process_noise', type=float, default=0.14)
    parser.add_argument('-vmn', '--var_measurement_noise', type=float, default=3)
    parser.add_argument('-t', '--threshold', type=float, default=0.4)
    parser.add_argument('-maf', '--max_active_filters', type=int, default=2)
    parser.add_argument('-mm', '--max_miss', type=int, default=6)
    parser.add_argument('-no_p', action='store_false', default=True)
    parser.add_argument('-s', '--save', default=False)
    parser.add_argument('-no_pk', action='store_false', default=True)
    parser.add_argument('-f', '--filter_short_frames', type=int, default=0)
    parser.add_argument('--frame_range', type=int, default=[35800, 36000], nargs='+')
    parser.add_argument('--plot-x-limits', type=float, nargs=2, default=(-3, 3))
    parser.add_argument('--plot-y-limits', type=float, nargs=2, default=(1, 6))
    parser.add_argument('-fn', '--folder_name', default='kalman')
    parser.add_argument('--dim', type=int, choices=[2, 3], default=2, help="Dimensions to use: 2 -> 'xy', 3 -> 'xyz'.")
    parser.add_argument('--kalman-pred-score-threshold', '--kalman_pred_score_threshold', type=float, default=0.5)
    parser.add_argument('--tracking-score-threshold', '--tracking_score_threshold', type=float, default=0.5)
    parser.add_argument('--tracking-match-threshold', '--tracking_match_threshold', type=float, default=1.0)
    parser.add_argument('--tracking-match-dims', '--tracking_match_dims', type=int, choices=[2, 3], default=2)
    parser.add_argument('--stabilize-gt', '--stabilize_gt', action='store_true')
    parser.add_argument('--gt-stabilization-threshold', '--gt_stabilization_threshold', type=float, default=0.2)
    parser.add_argument('--evaluation-y-limit', type=float, default=4.9, help='Apply the selected post-Kalman evaluation rule to frames where any GT position has y above this limit in meters.')
    parser.add_argument('--evaluation-y-limit-mode', choices=('ignore', 'cut-equalize', 'zero'), default='ignore', help='ignore skips out-of-range frames and resets association evaluation after each gap. cut-equalize reproduces limit_prediction_for_testset.py: remove out-of-range frames per recording and trim all recordings to the shortest remaining length. zero keeps the timeline and zeros frames.')
    parser.add_argument('--sequence-lengths', type=int, nargs='+', default=None, help='Frame counts per recording. By default, recording_sequence_lengths.npy is loaded from the input folder when present.')
    parser.add_argument('--kalman-reset-between-sequences', action='store_true', help='Reset Kalman state at recording boundaries. Without this flag, Kalman runs continuously to reproduce the historical evaluation pipeline.')
    parser.add_argument('--tracking-reset-between-sequences', action='store_true', help='Reset metric state and prediction track IDs at recording boundaries. Without this flag, reproduce the historical continuous evaluation.')
    args = parser.parse_args()
    if args.dim == 2:
        meas_idx = [0, 1]
    else:
        meas_idx = [0, 1, 2]
    pred_scores = torch.tensor(np.load(os.path.join(args.input, 'preds_classification.npy')), dtype=torch.float32)
    if pred_scores.ndim != 2:
        raise ValueError(f'preds_classification.npy must have shape [frames, people], got {tuple(pred_scores.shape)}.')
    pred_cls = (pred_scores >= float(args.kalman_pred_score_threshold)).to(torch.float32)
    pred_pos_and_velo = torch.tensor(np.load(os.path.join(args.input, 'preds_regression.npy')), dtype=torch.float32).reshape(pred_scores.shape[0], pred_scores.shape[1], -1)
    gt_reg = torch.tensor(np.load(os.path.join(args.input, 'targets_regression.npy')), dtype=torch.float32)
    gt_cls = torch.tensor(np.load(os.path.join(args.input, 'targets_classification.npy')), dtype=torch.float32)
    if gt_cls.ndim != 2:
        raise ValueError(f'targets_classification.npy must have shape [frames, people], got {tuple(gt_cls.shape)}.')
    if gt_reg.ndim == 2:
        if gt_reg.shape[1] % gt_cls.shape[1] != 0:
            raise ValueError(f'Flattened targets_regression.npy columns must be divisible by the number of GT people: {gt_reg.shape[1]} % {gt_cls.shape[1]} != 0.')
        gt_reg = gt_reg.reshape(gt_reg.shape[0], gt_cls.shape[1], -1)
    elif gt_reg.ndim != 3:
        raise ValueError(f'targets_regression.npy must have shape [frames, people, dimensions] or [frames, people * dimensions], got {tuple(gt_reg.shape)}.')
    recording_sequence_lengths = args.sequence_lengths
    sequence_lengths_path = os.path.join(args.input, 'recording_sequence_lengths.npy')
    if recording_sequence_lengths is None and os.path.isfile(sequence_lengths_path):
        recording_sequence_lengths = np.load(sequence_lengths_path).tolist()
    if recording_sequence_lengths is None:
        legacy_summary_path = os.path.join(args.input, f'parts3_filtered_limit_{args.evaluation_y_limit:g}_nocut', 'summary.json')
        if os.path.isfile(legacy_summary_path):
            with open(legacy_summary_path, encoding='utf-8') as file:
                legacy_summary = json.load(file)
            recording_sequence_lengths = [int(part['orig_len']) for part in legacy_summary.get('parts', [])]
            if not recording_sequence_lengths:
                recording_sequence_lengths = None
    recording_sequence_lengths = _normalize_sequence_lengths(recording_sequence_lengths, pred_pos_and_velo.shape[0])
    kalman_sequence_lengths = recording_sequence_lengths if args.kalman_reset_between_sequences else [int(pred_pos_and_velo.shape[0])]
    (preds_regression, preds_classification, pred_track_ids, active_filters, dead_filters_history, all_filters, _) = apply_kalman_per_sequence(pred_pos_and_velo, pred_scores, args.var_process_noise, args.var_measurement_noise, args.threshold, args.max_miss, args.max_active_filters, args.filter_short_frames, sequence_lengths=kalman_sequence_lengths, dt=1 / 15, dim=args.dim, pred_score_threshold=args.kalman_pred_score_threshold)
    gt_regression = gt_reg.numpy()
    gt_classification = gt_cls.numpy()
    gt_track_ids = None
    if args.stabilize_gt:
        gt_track_ids = infer_gt_track_ids_per_sequence(gt_regression, gt_classification >= 0.5, recording_sequence_lengths, match_dims=args.tracking_match_dims, stabilization_threshold=args.gt_stabilization_threshold)
    limit_summary = None
    evaluation_mask = np.ones(gt_regression.shape[0], dtype=bool)
    if args.evaluation_y_limit_mode == 'ignore':
        frames_over_y_limit_mask = (gt_regression[:, :, 1] > float(args.evaluation_y_limit)).any(axis=1)
        evaluation_mask = ~frames_over_y_limit_mask
        frames_over_y_limit = int(frames_over_y_limit_mask.sum())
        filtered_sequence_lengths = recording_sequence_lengths
    elif args.evaluation_y_limit_mode == 'cut-equalize':
        (preds_classification, preds_regression, gt_classification, gt_regression, filtered_sequence_lengths, limit_summary, pred_track_ids, gt_track_ids) = filter_sequences_and_equalize(preds_classification, preds_regression, gt_classification, gt_regression, recording_sequence_lengths, limit=args.evaluation_y_limit, pred_track_ids=pred_track_ids, gt_track_ids=gt_track_ids)
        frames_over_y_limit = sum((part['cut_rule1_limit'] for part in limit_summary['parts']))
        evaluation_mask = np.ones(gt_regression.shape[0], dtype=bool)
    else:
        (preds_classification, preds_regression, gt_classification, gt_regression, frames_over_y_limit_mask) = apply_y_limit(preds_classification, preds_regression, gt_classification, gt_regression, args.evaluation_y_limit, copy=False)
        frames_over_y_limit = int(frames_over_y_limit_mask.sum())
        filtered_sequence_lengths = recording_sequence_lengths
        pred_track_ids[preds_classification < args.tracking_score_threshold] = -1
        if gt_track_ids is not None:
            gt_track_ids[gt_classification < 0.5] = -1
        evaluation_mask = np.ones(gt_regression.shape[0], dtype=bool)
    if args.tracking_reset_between_sequences:
        sequence_breaks = np.cumsum(filtered_sequence_lengths)[:-1].tolist()
    else:
        sequence_breaks = []
    path = args.input
    os.makedirs(os.path.join(path, args.folder_name), exist_ok=True)
    np.save(os.path.join(path, args.folder_name, 'preds_regression.npy'), preds_regression)
    np.save(os.path.join(path, args.folder_name, 'preds_classification.npy'), preds_classification)
    np.save(os.path.join(path, args.folder_name, 'targets_regression.npy'), gt_regression)
    np.save(os.path.join(path, args.folder_name, 'targets_classification.npy'), gt_classification)
    np.save(os.path.join(path, args.folder_name, 'pred_track_ids.npy'), pred_track_ids)
    np.save(os.path.join(path, args.folder_name, 'evaluation_valid_mask.npy'), evaluation_mask)
    if gt_track_ids is not None:
        np.save(os.path.join(path, args.folder_name, 'gt_track_ids.npy'), gt_track_ids)
    np.save(os.path.join(path, args.folder_name, 'recording_sequence_lengths.npy'), np.asarray(filtered_sequence_lengths, dtype=np.int64))
    if limit_summary is not None:
        with open(os.path.join(path, args.folder_name, 'limit_summary.json'), 'w', encoding='utf-8') as file:
            json.dump(limit_summary, file, indent=2)
    tracking_metrics = compute_tracking_metrics_from_arrays(pred_regression=preds_regression, pred_scores=preds_classification, gt_regression=gt_regression, gt_valid=gt_classification >= 0.5, score_threshold=args.tracking_score_threshold, match_threshold=args.tracking_match_threshold, match_dims=args.tracking_match_dims, pred_track_ids=pred_track_ids, gt_track_ids=gt_track_ids, scores_are_logits=False, sequence_breaks=sequence_breaks, stabilize_gt=args.stabilize_gt and gt_track_ids is None, gt_stabilization_threshold=args.gt_stabilization_threshold, evaluation_mask=evaluation_mask)
    tracking_metrics['evaluation_mode'] = 'kalman_stateful'
    tracking_metrics['pred_track_id_mode'] = 'kalman_filter_ids'
    tracking_metrics['kalman_var_process_noise'] = float(args.var_process_noise)
    tracking_metrics['kalman_var_measurement_noise'] = float(args.var_measurement_noise)
    tracking_metrics['kalman_assignment_threshold'] = float(args.threshold)
    tracking_metrics['kalman_pred_score_threshold'] = float(args.kalman_pred_score_threshold)
    tracking_metrics['kalman_max_miss'] = int(args.max_miss)
    tracking_metrics['kalman_max_active_filters'] = int(args.max_active_filters)
    tracking_metrics['kalman_dim'] = int(args.dim)
    tracking_metrics['recording_sequence_lengths'] = filtered_sequence_lengths
    tracking_metrics['kalman_reset_between_sequences'] = bool(args.kalman_reset_between_sequences)
    tracking_metrics['tracking_reset_between_sequences'] = bool(args.tracking_reset_between_sequences)
    tracking_metrics['gt_track_id_mode'] = 'pre_filter_stabilized' if gt_track_ids is not None else 'slot_indices'
    tracking_metrics['evaluation_y_limit'] = float(args.evaluation_y_limit)
    tracking_metrics['evaluation_y_limit_mode'] = args.evaluation_y_limit_mode
    tracking_metrics['frames_over_evaluation_y_limit'] = int(frames_over_y_limit)
    save_tracking_metrics(tracking_metrics, os.path.join(path, args.folder_name, 'tracking_metrics.json'))
    print(f"Tracking metrics: F1={tracking_metrics['f1']:.4f}, MOTA={tracking_metrics['mota']:.4f}, HOTA={tracking_metrics['hota']:.4f}, IDSW={tracking_metrics['id_switches']}")
    gt_reg = torch.as_tensor(gt_regression)
    gt_cls = torch.as_tensor(gt_classification)
    start = int(args.frame_range[0])
    end = int(args.frame_range[1])
    if args.save == 'pdf':
        plt.close('all')
        plt.rcParams['font.family'] = 'Times New Roman'
        (fig, ax) = plt.subplots(figsize=(3.5, 2.8))
        plt.rcParams['axes.axisbelow'] = True
        ax.grid(True, zorder=0)
        ax.set_xlabel('x in m')
        ax.set_ylabel('y in m')
        ax.set_xlim(*args.plot_x_limits)
        ax.set_ylim(*args.plot_y_limits)
    else:
        (fig, ax) = plt.subplots()
        ax.grid(True, zorder=0)
        ax.set_xlabel('x in m')
        ax.set_ylabel('y in m')
        ax.set_xlim(*args.plot_x_limits)
        ax.set_ylim(*args.plot_y_limits)
        ax.set_aspect('equal', adjustable='box')
    if gt_reg.ndim == 2:
        try:
            gt_reg = gt_reg.numpy()
            gt_reg = gt_reg.reshape(gt_reg.shape[0], gt_cls.shape[1], 6)
        except:
            print('Error reshaping gt_reg to match preds_regression shape.')
            pass
    for i in range(gt_reg.shape[1]):
        valid = gt_cls[start:end, i] == 1
        if valid.sum().item() > 0:
            ax.scatter(gt_reg[start:end, i, 0][valid], gt_reg[start:end, i, 1][valid], c='blue', s=6, zorder=3)
    visible_predictions = []
    for i in range(pred_pos_and_velo.shape[1]):
        valid = (pred_cls[start:end, i] == 1) & (torch.abs(pred_pos_and_velo[start:end, i, :len(meas_idx)]).sum(dim=-1) > 0)
        if valid.sum().item() > 0:
            visible_predictions.append((i, valid))
    visible_filters = []
    for (idx, kf) in enumerate(all_filters):
        s = max(0, start - kf.start_frame)
        e = min(len(kf.history), end - kf.start_frame)
        if s < e:
            traj = torch.stack(kf.history[s:e])
            visible_filters.append((idx, traj))
    for (prediction_idx, valid) in visible_predictions:
        ax.scatter(pred_pos_and_velo[start:end, prediction_idx, 0][valid], pred_pos_and_velo[start:end, prediction_idx, 1][valid], color='red', s=6, zorder=3)
    if args.no_pk:
        track_colors = distinct_prediction_colors(len(visible_filters))
        for (color, (_, traj)) in zip(track_colors, visible_filters):
            ax.plot(traj[:, 0], traj[:, 1], color=color, zorder=4)
    circle_radius = args.threshold
    for kf in all_filters:
        s = max(0, start - kf.start_frame)
        e = min(len(kf.history), end - kf.start_frame)
        if s >= e:
            continue
        step = max(1, (e - s) // 8)
        for t_rel in range(s, e, step):
            x_t = kf.history[t_rel].view(6)
            pos_xy = x_t[0:2]
            circ = Ellipse(xy=(float(pos_xy[0]), float(pos_xy[1])), width=2 * circle_radius, height=2 * circle_radius, angle=0.0, fill=False, linestyle=':', linewidth=0.8, edgecolor='black')
    if args.save == 'pdf':
        plt.savefig(os.path.join(path, args.folder_name, f'{args.frame_range[0]}_{args.frame_range[1]}_kf_{args.var_process_noise}_{args.var_measurement_noise}_{args.threshold}_{args.max_miss}_{args.filter_short_frames}.pdf'), bbox_inches='tight')
    if args.save == 'png':
        plt.savefig(os.path.join(path, args.folder_name, f'{args.frame_range[0]}_{args.frame_range[1]}_kf_{args.var_process_noise}_{args.var_measurement_noise}_{args.threshold}_{args.max_miss}_{args.filter_short_frames}.png'))
    if args.no_p:
        plt.show()
