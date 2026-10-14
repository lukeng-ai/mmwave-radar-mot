import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
GT_COLORS = ['tab:blue', 'tab:gray', 'navy', 'dimgray']

def distinct_prediction_colors(count):
    """Return one deterministic, non-repeating color per prediction."""
    if count <= 0:
        return []
    hues = np.arange(count, dtype=float) / count
    hsv = np.column_stack((hues, np.full(count, 0.85), np.full(count, 0.8)))
    return hsv_to_rgb(hsv)

def load_score_threshold(input_path, fallback=0.3):
    metrics_path = os.path.join(input_path, 'metrics.json')
    if not os.path.isfile(metrics_path):
        return fallback
    try:
        with open(metrics_path, 'r', encoding='utf-8') as file:
            metrics = json.load(file)
        return float(metrics.get('score_threshold', fallback))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback

def flatten_time(array):
    if array.ndim == 4:
        return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])
    if array.ndim == 3:
        return array
    raise ValueError(f'Expected 3D or 4D array, got shape {array.shape}.')

def flatten_ids(array):
    if array.ndim == 3:
        return array.reshape(array.shape[0] * array.shape[1], array.shape[2])
    if array.ndim == 2:
        return array
    return None

def flatten_frame_matrix(array):
    if array.ndim == 3:
        return array.reshape(array.shape[0] * array.shape[1], array.shape[2])
    if array.ndim == 2:
        return array
    raise ValueError(f'Expected 2D or 3D array, got shape {array.shape}.')

def load_motr_predictions(input_path):
    npz_path = os.path.join(input_path, 'motr_sequence_predictions.npz')
    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        pred_regression = flatten_time(data['pred_regression'])
        pred_scores = flatten_frame_matrix(data['pred_scores'])
        gt_regression = flatten_time(data['gt_regression'])
        gt_valid = flatten_frame_matrix(data['gt_valid']).astype(bool)
        pred_track_ids = flatten_ids(data['pred_track_ids']) if 'pred_track_ids' in data.files else flatten_ids(data['selected_track_indices']) if 'selected_track_indices' in data.files else None
        evaluation_valid_mask = np.asarray(data['evaluation_valid_mask']).reshape(-1).astype(bool) if 'evaluation_valid_mask' in data.files else np.ones(pred_regression.shape[0], dtype=bool)
        return (pred_regression, pred_scores, gt_regression, gt_valid, pred_track_ids, evaluation_valid_mask)
    pred_regression = np.load(os.path.join(input_path, 'preds_regression.npy'))
    pred_scores = np.load(os.path.join(input_path, 'preds_classification.npy')).astype(float)
    gt_regression = np.load(os.path.join(input_path, 'targets_regression.npy'))
    gt_valid = np.load(os.path.join(input_path, 'targets_classification.npy')).astype(bool)
    return (pred_regression, pred_scores, gt_regression, gt_valid, None, np.ones(pred_regression.shape[0], dtype=bool))

def nonzero_xy(points):
    return np.abs(points[..., :2]).sum(axis=-1) > 0

def associate_gt_tracks(gt_regression, gt_valid, max_distance):
    track_ids = np.full(gt_valid.shape, -1, dtype=np.int64)
    active_tracks = {}
    next_track_id = 0
    for frame_idx in range(gt_regression.shape[0]):
        valid_indices = np.flatnonzero(gt_valid[frame_idx] & nonzero_xy(gt_regression[frame_idx]))
        if valid_indices.size == 0:
            active_tracks = {}
            continue
        current_positions = gt_regression[frame_idx, valid_indices, :2]
        assigned_tracks = set()
        assigned_detections = set()
        if active_tracks:
            active_ids = list(active_tracks.keys())
            active_positions = np.stack([active_tracks[track_id] for track_id in active_ids])
            distances = np.linalg.norm(active_positions[:, None, :] - current_positions[None, :, :], axis=-1)
            try:
                from scipy.optimize import linear_sum_assignment
                (row_indices, col_indices) = linear_sum_assignment(distances)
            except ImportError:
                (row_indices, col_indices) = (np.array([], dtype=int), np.array([], dtype=int))
            for (row_idx, col_idx) in zip(row_indices, col_indices):
                if distances[row_idx, col_idx] > max_distance:
                    continue
                track_id = active_ids[row_idx]
                det_idx = valid_indices[col_idx]
                track_ids[frame_idx, det_idx] = track_id
                assigned_tracks.add(track_id)
                assigned_detections.add(col_idx)
        for (col_idx, det_idx) in enumerate(valid_indices):
            if col_idx in assigned_detections:
                continue
            track_ids[frame_idx, det_idx] = next_track_id
            assigned_tracks.add(next_track_id)
            next_track_id += 1
        active_tracks = {int(track_ids[frame_idx, det_idx]): gt_regression[frame_idx, det_idx, :2] for det_idx in valid_indices if track_ids[frame_idx, det_idx] >= 0}
    return track_ids

def plot_tracks(args):
    (pred_regression, pred_scores, gt_regression, gt_valid, pred_track_ids, evaluation_valid_mask) = load_motr_predictions(args.input)
    plot_valid_mask = evaluation_valid_mask if args.respect_evaluation_mask else np.ones(pred_regression.shape[0], dtype=bool)
    score_threshold = float(args.score_threshold) if args.score_threshold is not None else load_score_threshold(args.input)
    start = max(0, int(args.frame_range[0]))
    end = min(int(args.frame_range[1]), pred_regression.shape[0])
    if end <= start:
        raise ValueError(f'Invalid frame_range {args.frame_range}; available frames: {pred_regression.shape[0]}.')
    os.makedirs(os.path.join(args.input, args.folder_name), exist_ok=True)
    (fig, ax) = plt.subplots()
    ax.grid(True, zorder=0)
    ax.set_xlabel('x in m')
    ax.set_ylabel('y in m')
    ax.set_xlim(*args.plot_x_limits)
    ax.set_ylim(*args.plot_y_limits)
    ax.set_aspect('equal', adjustable='box')
    gt_track_ids = associate_gt_tracks(gt_regression, gt_valid, max_distance=float(args.gt_association_threshold))
    for gt_track_id in sorted(np.unique(gt_track_ids[start:end])):
        if gt_track_id < 0:
            continue
        (frame_indices, gt_indices) = np.where((gt_track_ids[start:end] == gt_track_id) & plot_valid_mask[start:end, None])
        if frame_indices.size == 0:
            continue
        absolute_frames = frame_indices + start
        xy = gt_regression[absolute_frames, gt_indices, :2]
        valid = nonzero_xy(xy)
        if not np.any(valid):
            continue
        xy = xy[valid]
        color = GT_COLORS[int(gt_track_id) % len(GT_COLORS)]
        if not args.no_lines:
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.1, alpha=0.55, linestyle='--')
        ax.scatter(xy[:, 0], xy[:, 1], color=color, s=7, alpha=0.45, zorder=2, label=f'GT track {gt_track_id}')
    track_points = {}
    for frame_idx in range(start, end):
        if not plot_valid_mask[frame_idx]:
            continue
        for out_idx in range(pred_regression.shape[1]):
            if pred_scores[frame_idx, out_idx] < score_threshold:
                continue
            xy = pred_regression[frame_idx, out_idx, :2]
            if np.abs(xy).sum() <= 0:
                continue
            track_idx = out_idx
            if pred_track_ids is not None:
                track_idx = int(pred_track_ids[frame_idx, out_idx])
            if track_idx < 0:
                continue
            track_points.setdefault(track_idx, []).append(xy)
    sorted_tracks = sorted(track_points.items())
    prediction_colors = distinct_prediction_colors(len(sorted_tracks))
    for (color, (track_idx, points)) in zip(prediction_colors, sorted_tracks):
        xy = np.asarray(points)
        if not args.no_lines:
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.6, zorder=4)
        ax.scatter(xy[:, 0], xy[:, 1], color=color, s=9, zorder=5, label=f'TbA track {track_idx}')
    ax.legend(loc='best', fontsize=8)
    filename = f'{start}_{end}_motr_tracks_score_{score_threshold:g}.{args.save}'
    output_path = os.path.join(args.input, args.folder_name, filename)
    if args.save in ('png', 'pdf'):
        plt.savefig(output_path, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close(fig)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-s', '--save', default='png', choices=('png', 'pdf'))
    parser.add_argument('-no_p', dest='show', action='store_false', default=True)
    parser.add_argument('-fn', '--folder_name', default='motr_tracks')
    parser.add_argument('--frame_range', type=int, default=[0, 200], nargs='+')
    parser.add_argument('--plot-x-limits', type=float, nargs=2, default=(-3, 3))
    parser.add_argument('--plot-y-limits', type=float, nargs=2, default=(1, 6))
    parser.add_argument('--score-threshold', type=float, default=None)
    parser.add_argument('--gt-association-threshold', type=float, default=1.0)
    parser.add_argument('--no-lines', action='store_true')
    parser.add_argument('--respect-evaluation-mask', action='store_true', help='Hide frames excluded from tracking metrics. By default plots show the complete streaming inference, including frames beyond the y-limit.')
    plot_tracks(parser.parse_args())
