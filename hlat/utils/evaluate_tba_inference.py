import argparse
import json
import os
import sys
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset import Dataset
from model import *
import utils.config
from hlat.utils.prediction_outputs import save_preds_other_if_enabled
from hlat.utils.tracking_metrics import TrackingMetricsAccumulator

def unknown_constructor(loader, node):
    return str(node.value)
yaml.add_constructor(None, unknown_constructor, Loader=yaml.SafeLoader)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MOTR streaming inference over all timesteps. The model is run without GT matching; every newly promoted track receives a persistent prediction ID.')
    parser.add_argument('-c', '--config', default='./configs/tba_seq10_memory10.yaml')
    parser.add_argument('-i', '--input', default=None, help='Training/result folder containing config.yaml and best_model.pth.')
    parser.add_argument('-pdp', '--plot_data_path', default=None, help="Override cfg.DATASET['test'] with one recording path.")
    parser.add_argument('-sp', '--save_path_name', default='all')
    parser.add_argument('--checkpoint', default=None, help='Path to best_model.pth/model_states.pth.')
    parser.add_argument('-d', '--datasetdir', default=None)
    parser.add_argument('-s', '--slicesdir', default=None)
    parser.add_argument('-w', '--workdir', default=None)
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--score-threshold', type=float, default=None)
    parser.add_argument('--keep-track-score-threshold', type=float, default=None)
    parser.add_argument('--new-track-score-threshold', type=float, default=None)
    parser.add_argument('--max-inactive-steps', type=int, default=None)
    parser.add_argument('--promotion-suppression-radius', type=float, default=None)
    parser.add_argument('--match-threshold', type=float, default=1.0, help='Max distance for a prediction/GT pair to count as TP.')
    parser.add_argument('--match-dims', type=int, default=2, choices=(2, 3), help='Use xy or xyz distance for Hungarian matching and TP threshold.')
    parser.add_argument('--stabilize-gt', action='store_true', help='Stabilize GT identities by frame-to-frame GT position matching before tracking metrics.')
    parser.add_argument('--gt-stabilization-threshold', type=float, default=0.2, help='Max distance for GT identity stabilization. Defaults to 0.2 m.')
    parser.add_argument('--evaluation-y-limit', type=float, default=None, help='Ignore metric frames in which at least one valid GT target has y above this limit. MOTR state still advances through ignored frames.')
    parser.add_argument('--save-preds', action='store_true', help='Also save per-frame predictions and targets as motr_sequence_predictions.npz.')
    parser.add_argument('--window-batch-eval', action='store_true', help='Evaluate each dataset window independently. Default is streaming evaluation, which carries MOTR track_state across overlapping windows within each recording.')
    parser.add_argument('--context-replay', action='store_true', help='In streaming mode, replay every overlapping sequence window from the saved track_state before that window and evaluate only the newest frame.')
    parser.add_argument('--evaluate-warmup', action='store_true', help='Also evaluate warm-up frames from the first window. By default only one output per sliding window is evaluated/saved, matching generate_preds_files.py.')
    parser.add_argument('--no-slot-align', action='store_true', help='Do not align MOTR query slots to target slots before writing generate_results-compatible files.')
    return parser.parse_args()

def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

def get_motr_model(model):
    if callable(getattr(model, 'forward_frame', None)):
        return model
    wrapped_model = getattr(model, 'model', None)
    if callable(getattr(wrapped_model, 'forward_frame', None)):
        return wrapped_model
    return None

def clone_track_state(track_state):
    if track_state is None:
        return None
    return {key: value.detach().clone() if torch.is_tensor(value) else value for (key, value) in track_state.items()}

def get_default_score_threshold(cfg):
    for name in ('TRACKFORMER_KEEP_TRACK_SCORE_THRESH', 'TRACKFORMER_NEW_TRACK_SCORE_THRESH', 'MOTR_KEEP_TRACK_SCORE_THRESH', 'MOTR_NEW_TRACK_SCORE_THRESH', 'ADATRACK_KEEP_TRACK_SCORE_THRESH', 'ADATRACK_NEW_TRACK_SCORE_THRESH', 'BYTRACK_HIGH_SCORE_THRESH'):
        value = getattr(cfg, name, None)
        if value is not None:
            return float(value)
    return 0.5

def get_tracking_approach(model):
    return str(getattr(model, 'tracking_approach', '')).lower()

def ensure_streaming_track_state(model, track_state, frame):
    if track_state is not None:
        return track_state
    if get_tracking_approach(model) == 'trackformer' and callable(getattr(model, 'init_track_state', None)):
        return model.init_track_state(frame.shape[0], device=frame.device)
    return track_state

def assign_persistent_track_instance_ids(previous_track_state, frame_out):
    next_track_state = frame_out['track_state']
    active_mask = next_track_state['active_mask']
    (batch_size, num_queries) = active_mask.shape
    previous_ids = None
    next_track_id = 0
    if previous_track_state is not None:
        previous_ids = previous_track_state.get('_evaluation_track_instance_ids')
        next_track_id = int(previous_track_state.get('_evaluation_next_track_instance_id', 0))
    if previous_ids is None:
        previous_ids = torch.full((batch_size, num_queries), -1, device=active_mask.device, dtype=torch.long)
    else:
        previous_ids = previous_ids.to(device=active_mask.device).clone()
    track_instance_ids = previous_ids.masked_fill(~active_mask, -1)
    promoted_object_indices = frame_out.get('promoted_object_indices')
    promoted_mask = promoted_object_indices >= 0 if promoted_object_indices is not None else active_mask & (track_instance_ids < 0)
    newly_active_without_promotion = active_mask & (track_instance_ids < 0)
    new_instance_mask = promoted_mask | newly_active_without_promotion
    for (batch_idx, query_idx) in torch.nonzero(new_instance_mask, as_tuple=False).tolist():
        track_instance_ids[batch_idx, query_idx] = next_track_id
        next_track_id += 1
    selected_track_indices = frame_out.get('selected_track_indices')
    if selected_track_indices is None:
        output_track_ids = track_instance_ids
    else:
        output_track_ids = torch.gather(track_instance_ids, 1, selected_track_indices)
    next_track_state['_evaluation_track_instance_ids'] = track_instance_ids
    next_track_state['_evaluation_next_track_instance_id'] = next_track_id
    frame_out['pred_track_ids'] = output_track_ids
    return output_track_ids

def split_localization_target(target, cfg):
    num_people = int(cfg.PNT_NUM_PEOPLE)
    regression_dim = int(cfg.REGRESSION_PARAMETER)
    regression_source_dim = 6
    if target.shape[-1] < regression_source_dim * num_people:
        raise ValueError(f'Target has {target.shape[-1]} columns, but at least {regression_source_dim * num_people} are needed.')
    regression = target[..., :regression_source_dim * num_people]
    regression = regression.reshape(*target.shape[:-1], num_people, regression_source_dim)
    regression = regression[..., :regression_dim]
    fixed_cls_start = 30
    compact_cls_start = regression_source_dim * num_people
    if target.shape[-1] >= fixed_cls_start + num_people:
        cls_start = fixed_cls_start
    elif target.shape[-1] >= compact_cls_start + num_people:
        cls_start = compact_cls_start
    else:
        cls_start = None
    if cls_start is None:
        classification = torch.ones(*target.shape[:-1], num_people, device=target.device)
    else:
        classification = target[..., cls_start:cls_start + num_people]
    valid = classification >= 0.5
    return (regression, classification, valid)

def normalize_motr_aux(aux):
    regression = aux['pred_regression_per_frame']
    classification = aux['pred_classification_per_frame']
    if classification.dim() == regression.dim():
        classification = classification.squeeze(-1)
    if classification.dim() != 3:
        raise ValueError(f'Expected MOTR classification logits with shape [B, T, Q] or [B, T, Q, 1], got {tuple(classification.shape)}.')
    return (regression, classification)

def normalize_frame_output(frame_out):
    regression = frame_out['pred_regression']
    classification = frame_out['pred_classification']
    if classification.dim() == regression.dim():
        classification = classification.squeeze(-1)
    if classification.dim() != 2:
        raise ValueError(f'Expected frame classification logits with shape [B, Q] or [B, Q, 1], got {tuple(classification.shape)}.')
    return (regression, classification)

def mask_predictions_without_track_ids(pred_logits, pred_track_ids):
    """Prevent inactive query slots from being counted as detections."""
    if pred_track_ids is None:
        return pred_logits
    return pred_logits.masked_fill(pred_track_ids.to(device=pred_logits.device) < 0, -10.0)

def update_distance_stats(stats, pred_pos, gt_pos):
    diff = pred_pos - gt_pos
    xyz_error = float(torch.linalg.norm(diff[..., :min(3, diff.shape[-1])]).item())
    xy_error = float(torch.linalg.norm(diff[..., :min(2, diff.shape[-1])]).item())
    stats['sum_error_xyz'] += xyz_error
    stats['sum_sq_error_xyz'] += xyz_error ** 2
    stats['sum_error_xy'] += xy_error
    stats['sum_sq_error_xy'] += xy_error ** 2
    stats['matched_distance_count'] += 1

def evaluate_frame(pred_pos, pred_logits, gt_pos, gt_valid, score_threshold, match_threshold, match_dims):
    frame_stats = defaultdict(float)
    pred_scores = torch.sigmoid(pred_logits)
    pred_active = pred_scores >= score_threshold
    pred_indices = torch.nonzero(pred_active, as_tuple=False).flatten()
    gt_indices = torch.nonzero(gt_valid, as_tuple=False).flatten()
    frame_stats['num_predictions'] = int(pred_indices.numel())
    frame_stats['num_gt'] = int(gt_indices.numel())
    frame_stats['count_abs_error'] = abs(frame_stats['num_predictions'] - frame_stats['num_gt'])
    matches = []
    if pred_indices.numel() > 0 and gt_indices.numel() > 0:
        pred_match_pos = pred_pos[pred_indices, :match_dims]
        gt_match_pos = gt_pos[gt_indices, :match_dims]
        dist = torch.cdist(pred_match_pos, gt_match_pos, p=2).detach().cpu().numpy()
        (row_ind, col_ind) = linear_sum_assignment(dist)
        assigned_preds = set()
        assigned_gt = set()
        for (row, col) in zip(row_ind, col_ind):
            distance = float(dist[row, col])
            if distance > match_threshold:
                continue
            pred_idx = int(pred_indices[row].item())
            gt_idx = int(gt_indices[col].item())
            assigned_preds.add(pred_idx)
            assigned_gt.add(gt_idx)
            matches.append((pred_idx, gt_idx, distance))
            update_distance_stats(frame_stats, pred_pos[pred_idx], gt_pos[gt_idx])
        frame_stats['tp'] = len(matches)
        frame_stats['fp'] = int(pred_indices.numel()) - len(assigned_preds)
        frame_stats['fn'] = int(gt_indices.numel()) - len(assigned_gt)
    else:
        frame_stats['tp'] = 0
        frame_stats['fp'] = int(pred_indices.numel())
        frame_stats['fn'] = int(gt_indices.numel())
    return (frame_stats, matches)

def add_stats(total, frame_stats):
    for (key, value) in frame_stats.items():
        total[key] += value

def finalize_metrics(stats, total_sequences, total_frames, sequence_length, runtime_sec, score_threshold, match_threshold, match_dims):
    tp = stats['tp']
    fp = stats['fp']
    fn = stats['fn']
    precision = tp / (tp + fp + 1e-09)
    recall = tp / (tp + fn + 1e-09)
    f1 = 2 * precision * recall / (precision + recall + 1e-09)
    matched_count = stats['matched_distance_count']
    mean_xyz = stats['sum_error_xyz'] / matched_count if matched_count else None
    rmse_xyz = np.sqrt(stats['sum_sq_error_xyz'] / matched_count) if matched_count else None
    mean_xy = stats['sum_error_xy'] / matched_count if matched_count else None
    rmse_xy = np.sqrt(stats['sum_sq_error_xy'] / matched_count) if matched_count else None
    id_switch_denominator = stats['id_continuations']
    id_switch_rate = stats['id_switches'] / id_switch_denominator if id_switch_denominator else None
    return {'num_sequences': int(total_sequences), 'sequence_length': int(sequence_length), 'num_evaluated_frames': int(total_frames), 'score_threshold': float(score_threshold), 'match_threshold': float(match_threshold), 'match_dims': int(match_dims), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'mean_position_error_xyz': None if mean_xyz is None else float(mean_xyz), 'rmse_position_error_xyz': None if rmse_xyz is None else float(rmse_xyz), 'mean_position_error_xy': None if mean_xy is None else float(mean_xy), 'rmse_position_error_xy': None if rmse_xy is None else float(rmse_xy), 'mean_abs_count_error': float(stats['count_abs_error'] / max(total_frames, 1)), 'id_switches': int(stats['id_switches']), 'id_continuations': int(stats['id_continuations']), 'id_switch_rate': None if id_switch_rate is None else float(id_switch_rate), 'track_fragmentations': int(stats['track_fragmentations']), 'runtime_sec': float(runtime_sec), 'fps': float(total_frames / runtime_sec) if runtime_sec > 0 else None}

def summarize_score_array(values):
    if not values:
        return None
    array = np.concatenate(values, axis=0).reshape(-1)
    if array.size == 0:
        return None
    return {'min': float(np.min(array)), 'p05': float(np.percentile(array, 5)), 'p25': float(np.percentile(array, 25)), 'median': float(np.percentile(array, 50)), 'p75': float(np.percentile(array, 75)), 'p95': float(np.percentile(array, 95)), 'max': float(np.max(array)), 'mean': float(np.mean(array))}

def add_streaming_diagnostics(score_diagnostics, frame_out, pred_logits):
    output_scores = torch.sigmoid(pred_logits[0]).detach().cpu().numpy()
    score_diagnostics['output_scores'].append(output_scores)
    for threshold in (0.05, 0.1, 0.2, 0.3, 0.5):
        key = f"output_count_ge_{str(threshold).replace('.', '_')}"
        score_diagnostics[key].append(np.array([(output_scores >= threshold).sum()]))
    track_logits = frame_out.get('pred_classification_track')
    if track_logits is not None:
        score_diagnostics['track_scores'].append(torch.sigmoid(track_logits[0]).detach().cpu().numpy())
    object_logits = frame_out.get('pred_classification_object')
    if object_logits is not None:
        score_diagnostics['object_scores'].append(torch.sigmoid(object_logits[0]).detach().cpu().numpy())
    active_track_mask = frame_out.get('active_track_mask')
    if active_track_mask is not None:
        score_diagnostics['internal_active_track_count'].append(np.array([int(active_track_mask[0].detach().cpu().sum().item())]))
    output_active_track_mask = frame_out.get('output_active_track_mask')
    if output_active_track_mask is not None:
        score_diagnostics['output_active_track_count'].append(np.array([int(output_active_track_mask[0].detach().cpu().sum().item())]))
    promoted_object_indices = frame_out.get('promoted_object_indices')
    if promoted_object_indices is not None:
        score_diagnostics['promoted_object_count'].append(np.array([int((promoted_object_indices[0].detach().cpu() >= 0).sum().item())]))

def finalize_streaming_diagnostics(score_diagnostics):
    metrics = {}
    for key in ('output_scores', 'track_scores', 'object_scores'):
        summary = summarize_score_array(score_diagnostics[key])
        if summary is not None:
            metrics[f'diagnostic_{key}'] = summary
    for key in ('internal_active_track_count', 'output_active_track_count', 'promoted_object_count', 'output_count_ge_0_05', 'output_count_ge_0_1', 'output_count_ge_0_2', 'output_count_ge_0_3', 'output_count_ge_0_5'):
        if score_diagnostics[key]:
            values = np.concatenate(score_diagnostics[key], axis=0).reshape(-1)
            metrics[f'diagnostic_{key}_mean'] = float(np.mean(values))
            metrics[f'diagnostic_{key}_sum'] = int(np.sum(values))
            metrics[f'diagnostic_{key}_hist'] = {str(int(value)): int(count) for (value, count) in zip(*np.unique(values.astype(int), return_counts=True))}
    return metrics

def make_flat_prediction(regression, logits, cfg):
    return torch.cat((regression.reshape(regression.shape[0], -1), logits.reshape(logits.shape[0], -1)), dim=-1)

def make_legacy_prediction_files(predict_tensor, ground_truth_tensor, result_path, cfg, classification_threshold=0.5):
    os.makedirs(result_path, exist_ok=True)
    predict_tensor_reg = torch.zeros((predict_tensor.shape[0], cfg.PNT_NUM_PEOPLE, 6), dtype=torch.float32)
    regression = predict_tensor[:, 0:cfg.REGRESSION_PARAMETER * cfg.PNT_NUM_PEOPLE]
    regression = regression.reshape(-1, cfg.PNT_NUM_PEOPLE, cfg.REGRESSION_PARAMETER)
    predict_tensor_reg[:, :cfg.PNT_NUM_PEOPLE, 0:cfg.REGRESSION_PARAMETER] = regression.cpu()
    np.save(os.path.join(result_path, 'targets_regression.npy'), ground_truth_tensor[:, 0:6 * cfg.PNT_NUM_PEOPLE].reshape(-1, cfg.PNT_NUM_PEOPLE, 6).cpu().numpy())
    np.save(os.path.join(result_path, 'targets_classification.npy'), ground_truth_tensor[:, 30:30 + cfg.PNT_NUM_PEOPLE].cpu().numpy())
    np.save(os.path.join(result_path, 'preds_regression.npy'), predict_tensor_reg.numpy())
    np.save(os.path.join(result_path, 'preds_classification.npy'), (nn.Sigmoid()(predict_tensor[:, cfg.REGRESSION_PARAMETER * cfg.PNT_NUM_PEOPLE:cfg.REGRESSION_PARAMETER * cfg.PNT_NUM_PEOPLE + cfg.PNT_NUM_PEOPLE]).cpu().numpy() >= classification_threshold).astype(int))
    save_preds_other_if_enabled(result_path, predict_tensor, cfg)

def compute_config_losses(predict_tensor, ground_truth_tensor, cfg):
    import utils.train_utils
    losses_dict = {}
    ind_losses_dict = {}
    for loss_x in cfg.TEST_LOSSES:
        loss_value = utils.train_utils.get_losses(loss_x)(predict_tensor, ground_truth_tensor, config=cfg)
        if isinstance(loss_value, list):
            losses_dict.setdefault(loss_x, []).append(loss_value)
        else:
            losses_dict[loss_x] = losses_dict.get(loss_x, 0) + float(loss_value)
    for ind_loss_x in cfg.IND_TEST_LOSSES:
        loss_value = utils.train_utils.get_losses(ind_loss_x)(predict_tensor, ground_truth_tensor, config=cfg, item=True, ind_loss_x=ind_losses_dict)
        if not isinstance(loss_value, list):
            ind_losses_dict[ind_loss_x] = float(loss_value)
    return (losses_dict, ind_losses_dict)

def apply_generate_preds_postprocessing(predict_tensor, ground_truth_tensor, cfg):
    import utils.train_utils
    output = predict_tensor
    if getattr(cfg, 'ADAPTIONS_FLAG', False):
        for adaption in cfg.ADAPTIONS:
            output = utils.train_utils.get_adaptions(adaption)(output, ground_truth_tensor, config=cfg)
    return output

def align_prediction_slots_to_targets(predict_tensor, ground_truth_tensor, cfg, match_dims=2, max_match_distance=None):
    num_people = int(cfg.PNT_NUM_PEOPLE)
    reg_dim = int(cfg.REGRESSION_PARAMETER)
    reg_end = reg_dim * num_people
    cls_end = reg_end + num_people
    output = predict_tensor.clone()
    pred_reg = output[:, :reg_end].reshape(-1, num_people, reg_dim)
    pred_cls = output[:, reg_end:cls_end]
    gt_reg = ground_truth_tensor[:, :6 * num_people].reshape(-1, num_people, 6)[:, :, :reg_dim]
    gt_cls = ground_truth_tensor[:, 30:30 + num_people] >= 0.5
    aligned_reg = torch.zeros_like(pred_reg)
    aligned_cls = torch.full_like(pred_cls, -10.0)
    pos_dim = min(match_dims, reg_dim, gt_reg.shape[-1])
    for row_idx in range(pred_reg.shape[0]):
        gt_indices = torch.nonzero(gt_cls[row_idx], as_tuple=False).flatten()
        if gt_indices.numel() > 0:
            distances = torch.cdist(pred_reg[row_idx, :, :pos_dim].unsqueeze(0), gt_reg[row_idx, gt_indices, :pos_dim].unsqueeze(0), p=2).squeeze(0)
            (pred_assignment, gt_assignment) = linear_sum_assignment(distances.detach().cpu().numpy())
            for (pred_slot, gt_local_idx) in zip(pred_assignment, gt_assignment):
                distance = float(distances[pred_slot, gt_local_idx].item())
                if max_match_distance is not None and distance > max_match_distance:
                    continue
                gt_slot = int(gt_indices[gt_local_idx].item())
                aligned_reg[row_idx, gt_slot] = pred_reg[row_idx, pred_slot]
                aligned_cls[row_idx, gt_slot] = pred_cls[row_idx, pred_slot]
    output[:, :reg_end] = aligned_reg.reshape(-1, reg_end)
    output[:, reg_end:cls_end] = aligned_cls
    return output

def print_config_losses(losses_dict, ind_losses_dict, prefix='Test'):
    scalar_losses = [f'{prefix} Mean {name}: {value:.4f}' for (name, value) in losses_dict.items() if not isinstance(value, list)]
    scalar_ind_losses = [f'{prefix} Mean {name}: {value:.4f}' for (name, value) in ind_losses_dict.items() if not isinstance(value, list)]
    print(', '.join(scalar_losses + scalar_ind_losses))

def evaluate_motr(model, dataloader, cfg, score_threshold, match_threshold, match_dims, stabilize_gt=False, gt_stabilization_threshold=None, save_preds=False, evaluate_warmup=False):
    model.eval()
    stats = defaultdict(float)
    tracking_metrics = TrackingMetricsAccumulator(score_threshold=score_threshold, match_threshold=match_threshold, match_dims=match_dims, scores_are_logits=True, stabilize_gt=stabilize_gt, gt_stabilization_threshold=gt_stabilization_threshold)
    total_sequences = 0
    total_frames = 0
    saved = {'pred_regression': [], 'pred_scores': [], 'gt_regression': [], 'gt_valid': []}
    predict_rows = []
    ground_truth_rows = []
    start_time = time.time()
    with torch.no_grad():
        for (radar, target, *rest) in dataloader:
            radar = radar.to(cfg.DEVICE, non_blocking=True)
            target = target.to(cfg.DEVICE, non_blocking=True)
            rest = tuple((item.to(cfg.DEVICE, non_blocking=True) for item in rest))
            motr_input = rest[0] if rest else radar
            (_, aux) = model(motr_input, return_aux=True)
            (pred_regression, pred_logits) = normalize_motr_aux(aux)
            if target.dim() == 2:
                target = target.unsqueeze(1)
            (gt_regression, _, gt_valid) = split_localization_target(target, cfg)
            (batch_size, sequence_length) = pred_regression.shape[:2]
            if gt_regression.shape[:2] != (batch_size, sequence_length):
                raise RuntimeError(f'Prediction/target sequence shape mismatch: pred {tuple(pred_regression.shape[:2])}, target {tuple(gt_regression.shape[:2])}.')
            eval_indices = list(range(sequence_length)) if evaluate_warmup else [sequence_length - 1]
            total_sequences += batch_size
            total_frames += batch_size * len(eval_indices)
            previous_assignment = [dict() for _ in range(batch_size)]
            previous_seen = [set() for _ in range(batch_size)]
            for b in range(batch_size):
                tracking_metrics.reset_sequence()
                for t in eval_indices:
                    (frame_stats, matches) = tracking_metrics.update(pred_regression[b, t], pred_logits[b, t], gt_regression[b, t], gt_valid[b, t])
                    add_stats(stats, frame_stats)
                    current_assignment = {gt_idx: pred_idx for (pred_idx, gt_idx, _) in matches}
                    current_seen = set(current_assignment.keys())
                    for (gt_idx, pred_idx) in current_assignment.items():
                        if gt_idx in previous_assignment[b]:
                            stats['id_continuations'] += 1
                            if previous_assignment[b][gt_idx] != pred_idx:
                                stats['id_switches'] += 1
                    for gt_idx in previous_seen[b]:
                        if gt_idx not in current_seen and bool(gt_valid[b, t, gt_idx].item()):
                            stats['track_fragmentations'] += 1
                    previous_assignment[b] = current_assignment
                    previous_seen[b] = current_seen
            if save_preds:
                saved['pred_regression'].append(pred_regression[:, eval_indices].detach().cpu().numpy())
                saved['pred_scores'].append(torch.sigmoid(pred_logits[:, eval_indices]).detach().cpu().numpy())
                saved['gt_regression'].append(gt_regression[:, eval_indices].detach().cpu().numpy())
                saved['gt_valid'].append(gt_valid[:, eval_indices].detach().cpu().numpy())
            predict_rows.append(make_flat_prediction(pred_regression[:, eval_indices].reshape(batch_size * len(eval_indices), *pred_regression.shape[2:]), pred_logits[:, eval_indices].reshape(batch_size * len(eval_indices), pred_logits.shape[-1]), cfg).detach().cpu())
            ground_truth_rows.append(target[:, eval_indices].reshape(batch_size * len(eval_indices), -1).detach().cpu())
    runtime_sec = time.time() - start_time
    metrics = finalize_metrics(stats, total_sequences, total_frames, getattr(cfg, 'SEQUENCE_LENGTH', 1), runtime_sec, score_threshold, match_threshold, match_dims)
    metrics.update(tracking_metrics.finalize(num_sequences=total_sequences, sequence_length=getattr(cfg, 'SEQUENCE_LENGTH', 1), runtime_sec=runtime_sec))
    if save_preds:
        saved = {key: np.concatenate(value, axis=0) if value else np.empty(0) for (key, value) in saved.items()}
    else:
        saved = None
    predict_tensor = torch.cat(predict_rows, dim=0) if predict_rows else torch.empty(0)
    ground_truth_tensor = torch.cat(ground_truth_rows, dim=0) if ground_truth_rows else torch.empty(0)
    return (metrics, saved, predict_tensor, ground_truth_tensor)

def get_recording_sequence_counts(dataset, cfg):
    recordings = getattr(dataset, 'recordings_list', None)
    dataset_path = getattr(dataset, 'datasetPath', None)
    data_part = float(getattr(dataset, 'data_part', 1))
    sequence_length = int(getattr(cfg, 'SEQUENCE_LENGTH', 1))
    if not recordings or dataset_path is None:
        return [len(dataset)]
    sequence_counts = []
    for recording in recordings:
        labels_path = os.path.join(dataset_path + recording, getattr(cfg, 'DATA_TARGET_FILENAME', 'labels_final_kalman.npy'))
        n_frames = np.load(labels_path, mmap_mode='r').shape[0]
        n_sequences = max(0, n_frames - sequence_length + 1)
        n_sequences = int(np.floor(n_sequences * data_part))
        sequence_counts.append(n_sequences)
    if sum(sequence_counts) != len(dataset):
        print(f'Warning: inferred recording sequence counts do not match dataset length ({sum(sequence_counts)} != {len(dataset)}). Falling back to one continuous stream.')
        return [len(dataset)]
    return sequence_counts

def to_device_sample(sample, device):
    (radar, target, *rest) = sample
    radar = torch.as_tensor(radar).unsqueeze(0).to(device)
    target = torch.as_tensor(target).unsqueeze(0).to(device)
    rest = tuple((torch.as_tensor(item).unsqueeze(0).to(device) for item in rest))
    return (radar, target, rest)

def evaluate_motr_streaming(motr_model, dataset, cfg, score_threshold, match_threshold, match_dims, stabilize_gt=False, gt_stabilization_threshold=None, save_preds=False, evaluate_warmup=False, context_replay=False, evaluation_y_limit=None):
    motr_model.eval()
    stats = defaultdict(float)
    tracking_metrics = TrackingMetricsAccumulator(score_threshold=score_threshold, match_threshold=match_threshold, match_dims=match_dims, scores_are_logits=True, stabilize_gt=stabilize_gt, gt_stabilization_threshold=gt_stabilization_threshold)
    total_frames = 0
    evaluation_segments = 0
    ignored_frames = 0
    sequence_counts = get_recording_sequence_counts(dataset, cfg)
    saved = {'pred_regression': [], 'pred_scores': [], 'gt_regression': [], 'gt_valid': [], 'recording_index': [], 'frame_index': [], 'selected_track_indices': [], 'pred_track_ids': [], 'evaluation_valid_mask': []}
    predict_rows = []
    ground_truth_rows = []
    score_diagnostics = defaultdict(list)
    global_index = 0
    start_time = time.time()
    with torch.no_grad():
        for (recording_index, n_sequences) in enumerate(sequence_counts):
            track_state = None
            state_after_frame = []
            previous_assignment = {}
            previous_seen = set()
            pending_metric_reset = True
            for sequence_index in range(n_sequences):
                (radar, target, rest) = to_device_sample(dataset[global_index], cfg.DEVICE)
                global_index += 1
                motr_input = rest[0] if rest else radar
                if target.dim() == 2:
                    target = target.unsqueeze(1)
                if context_replay:
                    frames_to_run = motr_input
                    targets_to_eval = target
                    frame_offset = 0
                    frame_already_encoded = False
                    eval_start = 0 if sequence_index == 0 and evaluate_warmup else frames_to_run.shape[1] - 1
                    pre_window_state_index = sequence_index - 1
                    if pre_window_state_index >= 0:
                        track_state = clone_track_state(state_after_frame[pre_window_state_index])
                    else:
                        track_state = None
                elif sequence_index == 0:
                    frames_to_run = motr_input
                    targets_to_eval = target
                    frame_offset = 0
                    frame_already_encoded = False
                    eval_start = 0 if evaluate_warmup else frames_to_run.shape[1] - 1
                else:
                    frames_to_run = motr_input[:, -1:]
                    targets_to_eval = target[:, -1:]
                    frame_offset = int(getattr(cfg, 'SEQUENCE_LENGTH', 1)) - 1
                    frame_already_encoded = False
                    eval_start = 0
                (gt_regression, _, gt_valid) = split_localization_target(targets_to_eval, cfg)
                for t in range(frames_to_run.shape[1]):
                    frame = frames_to_run[:, t]
                    track_state = ensure_streaming_track_state(motr_model, track_state, frame)
                    previous_track_state = track_state
                    frame_out = motr_model.forward_frame(frame, track_queries=track_state, frame_already_encoded=frame_already_encoded)
                    track_state = frame_out['track_state']
                    pred_track_ids = assign_persistent_track_instance_ids(previous_track_state, frame_out)
                    absolute_frame_index = sequence_index + frame_offset + t
                    if context_replay:
                        while len(state_after_frame) <= absolute_frame_index:
                            state_after_frame.append(None)
                        state_after_frame[absolute_frame_index] = clone_track_state(track_state)
                    (pred_regression, pred_logits) = normalize_frame_output(frame_out)
                    pred_logits = mask_predictions_without_track_ids(pred_logits, pred_track_ids)
                    if t < eval_start:
                        continue
                    selected_track_indices = frame_out.get('selected_track_indices')
                    frame_is_valid = not (evaluation_y_limit is not None and bool((gt_valid[0, t] & (gt_regression[0, t, :, 1] > float(evaluation_y_limit))).any().item()))
                    add_streaming_diagnostics(score_diagnostics, frame_out, pred_logits)
                    predict_rows.append(make_flat_prediction(pred_regression, pred_logits, cfg).detach().cpu())
                    ground_truth_rows.append(targets_to_eval[:, t].detach().cpu())
                    if not frame_is_valid:
                        ignored_frames += 1
                        pending_metric_reset = True
                        previous_assignment = {}
                        previous_seen = set()
                    else:
                        if pending_metric_reset:
                            tracking_metrics.reset_sequence()
                            evaluation_segments += 1
                            pending_metric_reset = False
                        (frame_stats, matches) = tracking_metrics.update(pred_regression[0], pred_logits[0], gt_regression[0, t], gt_valid[0, t], pred_track_ids=pred_track_ids[0].detach().cpu())
                        add_stats(stats, frame_stats)
                        total_frames += 1
                        current_assignment = {gt_idx: int(pred_track_ids[0, pred_idx].item()) for (pred_idx, gt_idx, _) in matches}
                        current_seen = set(current_assignment.keys())
                        for (gt_idx, pred_idx) in current_assignment.items():
                            if gt_idx in previous_assignment:
                                stats['id_continuations'] += 1
                                if previous_assignment[gt_idx] != pred_idx:
                                    stats['id_switches'] += 1
                        for gt_idx in previous_seen:
                            if gt_idx not in current_seen and bool(gt_valid[0, t, gt_idx].item()):
                                stats['track_fragmentations'] += 1
                        previous_assignment = current_assignment
                        previous_seen = current_seen
                    if save_preds:
                        saved['pred_regression'].append(pred_regression.detach().cpu().numpy())
                        saved['pred_scores'].append(torch.sigmoid(pred_logits).detach().cpu().numpy())
                        saved['gt_regression'].append(gt_regression[:, t].detach().cpu().numpy())
                        saved['gt_valid'].append(gt_valid[:, t].detach().cpu().numpy())
                        saved['recording_index'].append(np.array([recording_index], dtype=np.int64))
                        saved['frame_index'].append(np.array([absolute_frame_index], dtype=np.int64))
                        selected_track_indices = frame_out.get('selected_track_indices')
                        if selected_track_indices is None:
                            selected_track_indices = torch.arange(pred_regression.shape[1], device=pred_regression.device, dtype=torch.long).unsqueeze(0)
                        saved['selected_track_indices'].append(selected_track_indices.detach().cpu().numpy())
                        saved['pred_track_ids'].append(pred_track_ids.detach().cpu().numpy())
                        saved['evaluation_valid_mask'].append(np.array([frame_is_valid], dtype=bool))
    runtime_sec = time.time() - start_time
    metrics = finalize_metrics(stats, evaluation_segments, total_frames, getattr(cfg, 'SEQUENCE_LENGTH', 1), runtime_sec, score_threshold, match_threshold, match_dims)
    metrics.update(tracking_metrics.finalize(num_sequences=evaluation_segments, sequence_length=getattr(cfg, 'SEQUENCE_LENGTH', 1), runtime_sec=runtime_sec))
    metrics['evaluation_mode'] = 'streaming_stateful_context_replay' if context_replay else 'streaming_stateful'
    metrics['tracking_approach'] = get_tracking_approach(motr_model)
    metrics['num_recordings'] = len(sequence_counts)
    metrics['num_ignored_frames'] = int(ignored_frames)
    metrics['evaluation_y_limit'] = None if evaluation_y_limit is None else float(evaluation_y_limit)
    metrics['evaluation_y_limit_mode'] = 'ignore'
    metrics['frames_over_evaluation_y_limit'] = int(ignored_frames)
    metrics['recording_sequence_counts'] = [int(count) for count in sequence_counts]
    metrics['keep_track_score_thresh'] = float(getattr(motr_model, 'keep_track_score_thresh', np.nan))
    metrics['new_track_score_thresh'] = float(getattr(motr_model, 'new_track_score_thresh', np.nan))
    metrics['max_inactive_steps'] = int(getattr(motr_model, 'max_inactive_steps', -1))
    metrics['promotion_suppression_radius'] = float(getattr(motr_model, 'promotion_suppression_radius', np.nan))
    metrics['motr_keep_track_score_thresh'] = metrics['keep_track_score_thresh']
    metrics['motr_new_track_score_thresh'] = metrics['new_track_score_thresh']
    metrics['motr_max_inactive_steps'] = metrics['max_inactive_steps']
    metrics['motr_promotion_suppression_radius'] = metrics['promotion_suppression_radius']
    metrics.update(finalize_streaming_diagnostics(score_diagnostics))
    if save_preds:
        saved = {key: np.concatenate(value, axis=0) if value else np.empty(0) for (key, value) in saved.items()}
    else:
        saved = None
    predict_tensor = torch.cat(predict_rows, dim=0) if predict_rows else torch.empty(0)
    ground_truth_tensor = torch.cat(ground_truth_rows, dim=0) if ground_truth_rows else torch.empty(0)
    return (metrics, saved, predict_tensor, ground_truth_tensor)

def main():
    args = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    config_path = args.config
    if args.input is not None and args.config == './configs/tba_seq10_memory10.yaml':
        config_path = os.path.join(args.input, 'config.yaml')
    cfg_args = {}
    if args.datasetdir is not None:
        cfg_args['DATA_PATH'] = args.datasetdir
    if args.slicesdir is not None:
        cfg_args['DATA_SLICE_PATH'] = args.slicesdir
    if args.workdir is not None:
        cfg_args['WORKDIR'] = args.workdir
    if args.batch_size is not None:
        cfg_args['batch_size'] = args.batch_size
    if args.num_workers is not None:
        cfg_args['num_workers'] = args.num_workers
    cfg = utils.config.Config(LOAD_FROM_CONFIG_FILE_FLAG=True, CONFIG_PATH=config_path, DEVICE=device, args=cfg_args)
    cfg.DEVICE = device
    if args.plot_data_path is not None:
        cfg.DATASET['test'] = [args.plot_data_path]
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint_base = args.input if args.input is not None else cfg.SAVE_PATH
        checkpoint = os.path.join(checkpoint_base, 'best_model.pth')
        if not os.path.isfile(checkpoint):
            fallback_checkpoint = os.path.join(checkpoint_base, 'model_states.pth')
            if os.path.isfile(fallback_checkpoint):
                print(f'Warning: best_model.pth not found. Falling back to {fallback_checkpoint}.')
                checkpoint = fallback_checkpoint
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}. Pass --checkpoint explicitly.')
    model_cls = globals()[cfg.MODEL]
    model = model_cls(config=cfg).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    motr_model = get_motr_model(model)
    if motr_model is None:
        raise RuntimeError(f'{cfg.MODEL} does not expose the MOTR/TrackFormer forward_frame API.')
    tracking_approach = get_tracking_approach(motr_model)
    if tracking_approach not in ('motr', 'trackformer', 'bytetrack'):
        raise RuntimeError(f"{cfg.MODEL} has tracking_approach={getattr(motr_model, 'tracking_approach', None)!r}; this evaluator is intended for stateful streaming tracking models.")
    if args.keep_track_score_threshold is not None:
        if tracking_approach == 'trackformer':
            cfg.TRACKFORMER_KEEP_TRACK_SCORE_THRESH = float(args.keep_track_score_threshold)
        else:
            cfg.MOTR_KEEP_TRACK_SCORE_THRESH = float(args.keep_track_score_threshold)
        motr_model.keep_track_score_thresh = float(args.keep_track_score_threshold)
    if args.new_track_score_threshold is not None:
        if tracking_approach == 'trackformer':
            cfg.TRACKFORMER_NEW_TRACK_SCORE_THRESH = float(args.new_track_score_threshold)
        else:
            cfg.MOTR_NEW_TRACK_SCORE_THRESH = float(args.new_track_score_threshold)
        motr_model.new_track_score_thresh = float(args.new_track_score_threshold)
    if args.max_inactive_steps is not None:
        cfg.MOTR_MAX_INACTIVE_STEPS = int(args.max_inactive_steps)
        motr_model.max_inactive_steps = int(args.max_inactive_steps)
    if args.promotion_suppression_radius is not None:
        cfg.MOTR_PROMOTION_SUPPRESSION_RADIUS = float(args.promotion_suppression_radius)
        motr_model.promotion_suppression_radius = float(args.promotion_suppression_radius)
    (_, _, test_dataset, _, _) = Dataset(config=cfg).load_dataset(only_test_set=True)
    score_threshold = float(args.score_threshold) if args.score_threshold is not None else get_default_score_threshold(cfg)
    if args.save_dir is not None:
        save_dir = args.save_dir
    elif args.input is not None:
        save_dir = os.path.join(args.input, args.save_path_name)
    else:
        save_dir = cfg.SAVE_PATH
    os.makedirs(save_dir, exist_ok=True)
    if args.window_batch_eval:
        test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, pin_memory=device.type == 'cuda', shuffle=False)
        (metrics, saved, predict_tensor, ground_truth_tensor) = evaluate_motr(model, test_loader, cfg, score_threshold=score_threshold, match_threshold=args.match_threshold, match_dims=args.match_dims, stabilize_gt=args.stabilize_gt, gt_stabilization_threshold=args.gt_stabilization_threshold, save_preds=args.save_preds, evaluate_warmup=args.evaluate_warmup)
        metrics['evaluation_mode'] = 'independent_windows'
    else:
        (metrics, saved, predict_tensor, ground_truth_tensor) = evaluate_motr_streaming(motr_model, test_dataset, cfg, score_threshold=score_threshold, match_threshold=args.match_threshold, match_dims=args.match_dims, stabilize_gt=args.stabilize_gt, gt_stabilization_threshold=args.gt_stabilization_threshold, save_preds=args.save_preds, evaluate_warmup=args.evaluate_warmup, context_replay=args.context_replay, evaluation_y_limit=args.evaluation_y_limit)
    if predict_tensor.numel() > 0:
        postprocessed_predict_tensor = apply_generate_preds_postprocessing(predict_tensor, ground_truth_tensor, cfg)
        if not args.no_slot_align:
            postprocessed_predict_tensor = align_prediction_slots_to_targets(postprocessed_predict_tensor, ground_truth_tensor, cfg, match_dims=args.match_dims, max_match_distance=args.match_threshold)
        make_legacy_prediction_files(postprocessed_predict_tensor, ground_truth_tensor, save_dir, cfg, classification_threshold=score_threshold)
        (losses_dict, ind_losses_dict) = compute_config_losses(postprocessed_predict_tensor, ground_truth_tensor, cfg)
        metrics['config_test_losses'] = losses_dict
        metrics['config_ind_test_losses'] = ind_losses_dict
        print_config_losses(losses_dict, ind_losses_dict)
    metrics_path = os.path.join(save_dir, 'motr_sequence_inference_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    legacy_metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(legacy_metrics_path, 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    if saved is not None:
        preds_path = os.path.join(save_dir, 'motr_sequence_predictions.npz')
        np.savez_compressed(preds_path, **saved)
    print(json.dumps(metrics, indent=2))
    print(f'Saved metrics to {metrics_path}')
    print(f'Saved generate_preds_files-compatible outputs to {save_dir}')
    if saved is not None:
        print(f'Saved predictions to {preds_path}')
if __name__ == '__main__':
    main()
