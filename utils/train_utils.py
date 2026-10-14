import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import timeit
import matplotlib.pyplot as plt
import os
import itertools
_PARTIAL_ASSIGNMENT_CACHE = {}

def _partial_assignment_candidates(num_rows, num_cols, device):
    device_key = (device.type, device.index)
    cache_key = (num_rows, num_cols, device_key)
    cached = _PARTIAL_ASSIGNMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    candidates = []
    for assignment in itertools.product(range(-1, num_rows), repeat=num_cols):
        assigned_rows = [row for row in assignment if row >= 0]
        if len(assigned_rows) == len(set(assigned_rows)):
            candidates.append(assignment)
    candidate_rows = torch.tensor(candidates, device=device, dtype=torch.long)
    candidate_mask = candidate_rows >= 0
    cached = (candidate_rows, candidate_mask)
    _PARTIAL_ASSIGNMENT_CACHE[cache_key] = cached
    return cached

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-d', '--datasetdir')
    parser.add_argument('-w', '--workdir')
    parser.add_argument('-s', '--slicesdir')
    parser.add_argument('-ramt', '--useramTrue', action='store_true')
    parser.add_argument('-ramf', '--useramFalse', dest='useramTrue', action='store_false')
    parser.set_defaults(useramTrue=True)
    parser.add_argument('-actiont', '--actionwiseTrue', action='store_true')
    parser.add_argument('-actionf', '--actionwiseFalse', dest='actionwiseTrue', action='store_false')
    parser.set_defaults(actionwiseTrue=False)
    (args, unknown_args) = parser.parse_known_args()
    extra_args = {}
    for i in range(0, len(unknown_args), 2):
        if i + 1 < len(unknown_args):
            key = unknown_args[i].lstrip('-')
            value = unknown_args[i + 1]
            extra_args[key] = value
    full_args = {**vars(args), **extra_args}
    return full_args

def log_params(list, logger):
    for param in list:
        logger.info(f'{param[0]}: {param[1]}')

def get_model_parameters(model):
    """
    Returns the total number of parameters and the number of trainable parameters in a PyTorch model.

    Parameters:
    model (torch.nn.Module): The PyTorch model.
    """
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return (total_params, trainable_params)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def printTimestamp(start, title, logger):
    stop = timeit.default_timer()
    run_time = stop - start
    (hours, rem) = divmod(run_time, 3600)
    (minutes, seconds) = divmod(rem, 60)
    print(f'{title}: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}')
    logger.info(f'{title}: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}')

def plot_and_save_train_val_loss(train_loss, val_loss, filename='_'):
    """
    Plots the training and validation loss and saves the plot as a file.
    
    Args:
        train_loss (list or array): Loss values for training.
        val_loss (list or array): Loss values for validation.
        filename (str): Filename for the saved plot. Default is 'train_val_loss_plot.png'.
    """
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, 'b.-', label='Training loss')
    plt.plot(epochs, val_loss, 'r.-', label='Validation loss')
    plt.title('Training and Validation Error over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Test loss')
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.savefig(os.path.join(filename, 'train_val_error_plot.png'), dpi=300)

def plot_and_save_test_loss(test_loss, filename='_', label='Test loss'):
    """
    Plots the training and validation loss and saves the plot as a file.
    
    Args:
        train_loss (list or array): Loss values for training.
        val_loss (list or array): Loss values for validation.
        filename (str): Filename for the saved plot. Default is 'train_val_loss_plot.png'.
    """
    epochs = range(1, len(test_loss) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, test_loss, 'g.-', label='Test')
    plt.title('Testing Error over Steps')
    plt.xlabel('Step')
    plt.ylabel(f'{label}')
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.savefig(os.path.join(filename, 'test_error_plot.png'), dpi=300)

def sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config, model=None):
    preds = predictions_regression.view(-1, config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER)
    labels = labels_regression.view(-1, config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER)
    preds_cls = predictions_classification.view(-1, config.PNT_NUM_PEOPLE)
    if config.PNT_NUM_PEOPLE <= 1:
        sorted_preds = preds
        sorted_preds_cls = preds_cls
    else:
        position_dims = _localization_regression_loss_dims(config, model=model, available_dims=config.REGRESSION_PARAMETER)
        perms = torch.tensor(list(itertools.permutations(range(config.PNT_NUM_PEOPLE))), device=preds.device, dtype=torch.long)
        permuted_pred_positions = preds[:, perms, :position_dims]
        label_positions = labels[:, :, :position_dims].unsqueeze(1)
        costs = torch.norm(permuted_pred_positions - label_positions, dim=-1).sum(dim=-1)
        best_perms = perms[torch.argmin(costs, dim=1)]
        batch_indices = torch.arange(preds.shape[0], device=preds.device).unsqueeze(1)
        sorted_preds = preds[batch_indices, best_perms]
        sorted_preds_cls = preds_cls[batch_indices, best_perms]
    sorted_preds = sorted_preds.reshape(-1, config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE)
    sorted_labels = labels.view(-1, config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE)
    sorted_labels_cls = labels_classification.view(-1, config.PNT_NUM_PEOPLE)
    return (sorted_preds, sorted_preds_cls, sorted_labels, sorted_labels_cls)

def _use_frame_position_sort(config, kwargs):
    if not getattr(config, 'LOC_LOSS_SORT_BY_POSITION_FLAG', False):
        return False
    trackformer_aux = kwargs.get('trackformer_aux')
    if trackformer_aux is not None:
        if trackformer_aux.get('sequence_slot_matching_applied', False):
            return False
        if trackformer_aux.get('gt_propagation_matching_applied', False):
            return False
    return True

def get_criterion(criterion_str='MSELossPosRegBCEClsMask'):
    criterions = {'MSELossPosRegBCEClsMask': calc_mse_position_regression_and_bce_classification_mask, 'MSELossPosRegSoftF1ClsMask': calc_mse_position_regression_and_soft_f1_classification_mask, 'MSELossPosRegSoftBAClsMask': calc_mse_position_regression_and_ba_classification_mask, 'BCECls': calc_bce_classification_loss}
    return criterions[criterion_str]

def get_losses(loss_str='MPE'):
    losses = {'MSELossPosRegBCEClsMask': calc_mse_position_regression_and_bce_classification_mask, 'MSELossPosRegSoftF1ClsMask': calc_mse_position_regression_and_soft_f1_classification_mask, 'MSELossPosRegSoftBAClsMask': calc_mse_position_regression_and_ba_classification_mask, 'BCECls': calc_bce_classification_loss, 'BACls': calc_balanced_accuracy_classification_loss, 'SoftBACls': calc_soft_balanced_accuracy_classification_loss, 'AccCls': calc_accuracy_classification_loss, 'F1Cls': calc_f1_classification_loss, 'SoftF1Cls': calc_soft_f1_loss_classification, 'SoftDetACls': calc_soft_deta_classification_loss, 'SoftPrecisionCls': calc_soft_precision_classification_loss, 'RecallCls': calc_recall_classification_loss, 'PrecisionCls': calc_precision_classification_loss, 'MSELossReg': calc_mse_regression_loss, 'MPE': calc_mean_position_error, 'MedPE': calc_median_position_error, 'MPEXyOnly': calc_mean_position_error_xy_only, 'MVE': calc_mean_velocity_error}
    return losses[loss_str]

def get_adaptions(adaption_str='SortByClosestPosition'):
    adaptions = {'SortByClosestPosition': sort_output_by_closest_position, 'SortByClosestPositionMultipleInputs': sort_output_by_closest_position_multiple_inputs, 'ApplyPositionDependentXyOffset': apply_position_dependent_xy_offset, 'ApplyCorrectClassification': apply_correct_classification}
    return adaptions[adaption_str]

def _masked_regression_loss_or_zero(loss_function, predictions, labels):
    if predictions.numel() == 0 or labels.numel() == 0:
        return predictions.sum() * 0.0
    return loss_function(predictions, labels)

def _configured_num_people(config):
    return int(getattr(config, 'PNT_NUM_PEOPLE', 2))

def _configured_gt_num_people(config):
    num_people = _configured_num_people(config)
    gt_num_people = getattr(config, 'PNT_GT_NUM_PEOPLE', None)
    if gt_num_people is None:
        return num_people
    return max(0, min(num_people, int(gt_num_people)))

def extract_localization_targets(ground_truth, config):
    num_people = _configured_num_people(config)
    gt_num_people = _configured_gt_num_people(config)
    reg_dim = int(getattr(config, 'REGRESSION_PARAMETER', 3))
    labels_regression = ground_truth.new_zeros(ground_truth.shape[0], num_people, reg_dim)
    labels_classification = ground_truth.new_zeros(ground_truth.shape[0], num_people)
    available_reg_slots = min(gt_num_people, ground_truth.shape[1] // 6)
    available_cls_slots = 0
    if ground_truth.shape[1] > 30:
        available_cls_slots = min(gt_num_people, ground_truth.shape[1] - 30)
    copy_reg_slots = min(num_people, available_reg_slots)
    copy_cls_slots = min(num_people, available_cls_slots)
    if copy_reg_slots > 0:
        labels_regression[:, :copy_reg_slots] = ground_truth[:, :6 * copy_reg_slots].view(-1, copy_reg_slots, 6)[:, :, :reg_dim]
    if copy_cls_slots > 0:
        labels_classification[:, :copy_cls_slots] = ground_truth[:, 30:30 + copy_cls_slots]
    return (labels_regression, labels_classification)

def extract_localization_components(output, ground_truth, config):
    num_people = _configured_num_people(config)
    reg_dim = int(getattr(config, 'REGRESSION_PARAMETER', 3))
    reg_cols = reg_dim * num_people
    cls_end = reg_cols + num_people
    predictions_regression = output[:, :reg_cols]
    predictions_classification = output[:, reg_cols:cls_end]
    (labels_regression, labels_classification) = extract_localization_targets(ground_truth, config)
    return (predictions_regression, predictions_classification, labels_regression.reshape(-1, reg_cols), labels_classification)

def bce_classification_loss_with_eos_weight(predictions_classification, labels_classification, config, loss_function_classification=F.binary_cross_entropy_with_logits):
    eos_coef = float(getattr(config, 'MOTR_EOS_COEF', 1.0))
    num_people = _configured_num_people(config)
    gt_num_people = _configured_gt_num_people(config)
    if eos_coef == 1.0 or gt_num_people >= num_people:
        return loss_function_classification(predictions_classification, labels_classification)
    element_loss = F.binary_cross_entropy_with_logits(predictions_classification, labels_classification.float(), reduction='none')
    slot_indices = torch.arange(labels_classification.shape[-1], device=labels_classification.device)
    padded_slot_mask = slot_indices >= gt_num_people
    weights = torch.ones_like(labels_classification, dtype=element_loss.dtype)
    weights = torch.where(padded_slot_mask.unsqueeze(0), torch.full_like(weights, eos_coef), weights)
    return (element_loss * weights).mean()

def soft_deta_classification_loss(predictions_classification, labels_classification, fp_weight=1.0, fn_weight=1.0, eps=1e-06):
    probabilities = torch.sigmoid(predictions_classification)
    labels = labels_classification.float()
    soft_tp = (probabilities * labels).sum(dim=-1)
    soft_fp = (probabilities * (1.0 - labels)).sum(dim=-1)
    soft_fn = ((1.0 - probabilities) * labels).sum(dim=-1)
    soft_deta = (soft_tp + eps) / (soft_tp + fp_weight * soft_fp + fn_weight * soft_fn + eps)
    return (1.0 - soft_deta).mean()

def _calc_temporal_presence_loss(output, trackformer_aux, config):
    weight = float(getattr(config, 'TEMPORAL_PRESENCE_LOSS_WEIGHT', 0.0))
    if weight <= 0.0 or trackformer_aux is None:
        return output.new_tensor(0.0)
    predictions = trackformer_aux.get('pred_classification_per_frame')
    target_sequence = trackformer_aux.get('target_sequence')
    if predictions is None or target_sequence is None or predictions.shape[1] < 2:
        return output.new_tensor(0.0)
    flat_targets = target_sequence.reshape(-1, target_sequence.shape[-1])
    (_, labels) = extract_localization_targets(flat_targets, config)
    labels = labels.reshape(target_sequence.shape[0], target_sequence.shape[1], -1)
    num_slots = min(predictions.shape[-1], labels.shape[-1])
    probabilities = torch.sigmoid(predictions[..., :num_slots])
    labels = labels[..., :num_slots]
    persistent = (labels[:, 1:] > 0.5) & (labels[:, :-1] > 0.5)
    if not bool(persistent.any()):
        return probabilities.sum() * 0.0
    probability_change = (probabilities[:, 1:] - probabilities[:, :-1]).abs()
    temporal_loss = probability_change[persistent].mean()
    return weight * temporal_loss

def _calc_motr_newborn_query_loss(output, trackformer_aux, config, reuse_cached=False):
    weight = float(getattr(config, 'MOTR_NEWBORN_LOSS_WEIGHT', 0.0))
    if weight <= 0 or trackformer_aux is None:
        return output.new_tensor(0.0)
    if reuse_cached and 'motr_newborn_loss' in trackformer_aux:
        return trackformer_aux['motr_newborn_loss']
    object_regression = trackformer_aux.get('pred_regression_object_per_frame')
    object_classification = trackformer_aux.get('pred_classification_object_per_frame')
    selected_track_mask = trackformer_aux.get('qim_selected_track_mask_per_frame')
    sequence_slot_indices = trackformer_aux.get('sequence_slot_indices')
    target_sequence = trackformer_aux.get('target_sequence')
    if object_regression is None or object_classification is None or target_sequence is None:
        return output.new_tensor(0.0)
    if object_classification.dim() == object_regression.dim():
        object_classification = object_classification.squeeze(-1)
    (batch_size, seq_length, num_object_queries) = object_classification.shape
    flat_targets = target_sequence.reshape(batch_size * seq_length, -1)
    (gt_regression, gt_classification) = extract_localization_targets(flat_targets, config)
    gt_regression = gt_regression.reshape(batch_size, seq_length, gt_regression.shape[-2], gt_regression.shape[-1])
    gt_classification = gt_classification.reshape(batch_size, seq_length, gt_classification.shape[-1])
    pos_dim = min(int(getattr(config, 'MOTR_NEWBORN_MATCH_DIMS', 2)), object_regression.shape[-1], gt_regression.shape[-1])
    match_cost_pos = float(getattr(config, 'MOTR_NEWBORN_MATCH_COST_POS', getattr(config, 'MATCH_COST_POS', 1.0)))
    match_cost_cls = float(getattr(config, 'MOTR_NEWBORN_MATCH_COST_CLS', getattr(config, 'MATCH_COST_CLS', 0.1)))
    eos_coef = float(getattr(config, 'MOTR_NEWBORN_EOS_COEF', getattr(config, 'MOTR_EOS_COEF', 0.1)))
    regression_weight = float(getattr(config, 'MOTR_NEWBORN_REGRESSION_WEIGHT', config.WEIGHT_REGRESSION))
    classification_weight = float(getattr(config, 'MOTR_NEWBORN_CLASSIFICATION_WEIGHT', config.WEIGHT_CLASSIFICATION))
    current_active = gt_classification > 0.5
    previous_active = torch.cat((torch.zeros_like(current_active[:, :1]), current_active[:, :-1]), dim=1)
    uncovered_gt = current_active & ~previous_active
    if selected_track_mask is not None and sequence_slot_indices is not None:
        selected_track_mask = selected_track_mask.to(current_active.device)
        num_track_slots = selected_track_mask.shape[-1]
        track_slots = sequence_slot_indices.to(device=current_active.device, dtype=torch.long)
        valid_track_slots = (track_slots >= 0) & (track_slots < num_track_slots)
        if num_track_slots > 0:
            expanded_track_slots = track_slots[:, None, :].expand(-1, seq_length, -1)
            selected_for_gt = selected_track_mask.gather(2, expanded_track_slots.clamp(0, num_track_slots - 1))
            covered_by_track = valid_track_slots[:, None, :] & selected_for_gt
        else:
            covered_by_track = torch.zeros_like(current_active)
        uncovered_gt = current_active & (~previous_active | ~covered_by_track)
    zero = object_regression.sum() * 0.0
    if num_object_queries == 0:
        regression_loss = zero
        classification_loss = zero
        matched_count = torch.zeros((), device=object_regression.device, dtype=torch.long)
    else:
        position_cost = torch.linalg.vector_norm(object_regression[..., None, :pos_dim] - gt_regression[..., None, :, :pos_dim], dim=-1)
        positive_classification_cost = -F.logsigmoid(object_classification).unsqueeze(-1)
        cost = (match_cost_pos * position_cost + match_cost_cls * positive_classification_cost).detach()
        (candidate_rows, candidate_mask) = _partial_assignment_candidates(num_object_queries, gt_classification.shape[-1], object_regression.device)
        gt_indices = torch.arange(gt_classification.shape[-1], device=object_regression.device)
        candidate_costs = cost[:, :, candidate_rows.clamp_min(0), gt_indices]
        candidate_costs = (candidate_costs * candidate_mask).sum(dim=-1)
        valid_candidates = (candidate_mask[None, None, :, :] == uncovered_gt[:, :, None, :]).all(dim=-1)
        candidate_costs = candidate_costs.masked_fill(~valid_candidates, torch.inf)
        best_candidates = torch.argmin(candidate_costs, dim=-1)
        matched_rows = candidate_rows[best_candidates]
        matched_mask = matched_rows >= 0
        safe_matched_rows = matched_rows.clamp_min(0)
        classification_target = torch.zeros_like(object_classification)
        classification_target.scatter_add_(2, safe_matched_rows, matched_mask.to(classification_target.dtype))
        regression_indices = safe_matched_rows.unsqueeze(-1).expand(-1, -1, -1, object_regression.shape[-1])
        matched_regression = object_regression.gather(2, regression_indices)
        squared_error = (matched_regression - gt_regression).square().sum(dim=-1)
        matches_per_frame = matched_mask.sum(dim=-1)
        regression_loss_per_frame = (squared_error * matched_mask).sum(dim=-1) / (matches_per_frame.clamp_min(1) * object_regression.shape[-1])
        frames_with_matches = matches_per_frame > 0
        regression_loss = (regression_loss_per_frame * frames_with_matches).sum() / frames_with_matches.sum().clamp_min(1)
        element_classification_loss = F.binary_cross_entropy_with_logits(object_classification, classification_target, reduction='none')
        classification_weights = torch.full_like(element_classification_loss, eos_coef)
        classification_weights = torch.where(classification_target > 0, torch.ones_like(classification_weights), classification_weights)
        classification_loss = ((element_classification_loss * classification_weights).sum(dim=-1) / classification_weights.sum(dim=-1).clamp_min(1e-06)).mean()
        matched_count = matched_mask.sum().detach()
    newborn_loss = weight * (regression_weight * regression_loss + classification_weight * classification_loss)
    trackformer_aux['motr_newborn_matched_count'] = matched_count
    trackformer_aux['motr_newborn_regression_loss'] = regression_loss.detach()
    trackformer_aux['motr_newborn_classification_loss'] = classification_loss.detach()
    trackformer_aux['motr_newborn_loss'] = newborn_loss.detach()
    return newborn_loss

def _is_slot_head_model(config, model=None):
    slot_head_model_names = {'pntkpconvtbd'}
    if str(getattr(config, 'MODEL', '')).lower() in slot_head_model_names:
        return True
    if model is not None:
        return model.__class__.__name__.lower() in slot_head_model_names
    return False

def _localization_regression_loss_dims(config, model=None, available_dims=3):
    configured_dims = getattr(config, 'LOCALIZATION_REGRESSION_LOSS_DIMS', None)
    if configured_dims is None:
        configured_dims = 2 if _is_slot_head_model(config, model=model) else 3
    return max(1, min(int(configured_dims), int(available_dims)))

def _localization_position_tensors(predictions_regression, labels_regression, config, model=None):
    regression_dim = int(getattr(config, 'REGRESSION_PARAMETER', 3))
    position_dims = _localization_regression_loss_dims(config, model=model, available_dims=regression_dim)
    predictions = predictions_regression.reshape(-1, config.PNT_NUM_PEOPLE, regression_dim)[..., :position_dims]
    labels = labels_regression.reshape(-1, config.PNT_NUM_PEOPLE, regression_dim)[..., :position_dims]
    return (predictions, labels, position_dims)

def _slot_matching_num_slots(config, model=None):
    if model is not None and hasattr(model, 'num_slots'):
        return int(model.num_slots)
    num_slots = getattr(config, 'PNT_NUM_SLOTS', None)
    if num_slots is None:
        num_slots = getattr(config, 'PNT_PRED_NUM_SLOTS', getattr(config, 'PNT_NUM_PEOPLE', 2))
    return int(num_slots)

def _extract_slot_head_all_predictions(output, config, model=None):
    num_people = _configured_num_people(config)
    reg_dim = int(getattr(config, 'REGRESSION_PARAMETER', 3))
    num_slots = _slot_matching_num_slots(config, model=model)
    base_cols = num_people * reg_dim + num_people
    all_reg_cols = num_slots * reg_dim
    all_cls_cols = num_slots
    all_reg_start = base_cols
    all_reg_end = all_reg_start + all_reg_cols
    all_cls_end = all_reg_end + all_cls_cols
    if output.shape[1] < all_cls_end:
        return (None, None)
    all_regression = output[:, all_reg_start:all_reg_end].reshape(output.shape[0], num_slots, reg_dim)
    all_classification = output[:, all_reg_end:all_cls_end]
    return (all_regression, all_classification)

def _calc_slot_head_matching_loss(output, ground_truth, config, model=None, loss_function_regression=nn.MSELoss()):
    if not _is_slot_head_model(config, model=model):
        return output.sum() * 0.0
    weight = getattr(config, 'SLOT_MATCHING_LOSS_WEIGHT', None)
    if weight is None:
        weight = 1.0
    weight = float(weight)
    if weight <= 0.0:
        return output.sum() * 0.0
    (all_regression, all_classification) = _extract_slot_head_all_predictions(output, config, model=model)
    if all_regression is None:
        return output.sum() * 0.0
    (labels_regression, labels_classification) = extract_localization_targets(ground_truth, config)
    num_slots = all_classification.shape[1]
    num_people = labels_classification.shape[1]
    if num_slots == 0 or num_people == 0:
        return output.sum() * 0.0
    match_dims = min(int(getattr(config, 'SLOT_MATCHING_MATCH_DIMS', 2)), all_regression.shape[-1], labels_regression.shape[-1])
    match_cost_pos = float(getattr(config, 'SLOT_MATCHING_MATCH_COST_POS', 1.0))
    match_cost_cls = float(getattr(config, 'SLOT_MATCHING_MATCH_COST_CLS', 0.1))
    eos_coef = float(getattr(config, 'SLOT_MATCHING_EOS_COEF', getattr(config, 'MOTR_EOS_COEF', 0.1)))
    regression_weight = float(getattr(config, 'SLOT_MATCHING_REGRESSION_WEIGHT', getattr(config, 'WEIGHT_REGRESSION', 1.0)))
    classification_weight = float(getattr(config, 'SLOT_MATCHING_CLASSIFICATION_WEIGHT', getattr(config, 'WEIGHT_CLASSIFICATION', 1.0)))
    position_cost = torch.linalg.vector_norm(all_regression[:, :, None, :match_dims] - labels_regression[:, None, :, :match_dims], dim=-1)
    positive_classification_cost = -F.logsigmoid(all_classification).unsqueeze(-1)
    cost = (match_cost_pos * position_cost + match_cost_cls * positive_classification_cost).detach()
    (candidate_rows, candidate_mask) = _partial_assignment_candidates(num_slots, num_people, all_regression.device)
    gt_indices = torch.arange(num_people, device=all_regression.device)
    candidate_costs = cost[:, candidate_rows.clamp_min(0), gt_indices]
    candidate_costs = (candidate_costs * candidate_mask).sum(dim=-1)
    active_gt = labels_classification > 0.5
    valid_candidates = (candidate_mask[None, :, :] == active_gt[:, None, :]).all(dim=-1)
    candidate_costs = candidate_costs.masked_fill(~valid_candidates, torch.inf)
    best_candidates = torch.argmin(candidate_costs, dim=-1)
    matched_rows = candidate_rows[best_candidates]
    matched_mask = matched_rows >= 0
    safe_matched_rows = matched_rows.clamp_min(0)
    classification_target = torch.zeros_like(all_classification)
    classification_target.scatter_add_(1, safe_matched_rows, matched_mask.to(classification_target.dtype))
    classification_target = classification_target.clamp_max(1.0)
    regression_indices = safe_matched_rows.unsqueeze(-1).expand(-1, -1, all_regression.shape[-1])
    matched_regression = all_regression.gather(1, regression_indices)
    regression_dims = _localization_regression_loss_dims(config, model=model, available_dims=min(all_regression.shape[-1], labels_regression.shape[-1]))
    regression_mask = matched_mask.unsqueeze(-1).expand(-1, -1, regression_dims)
    regression_loss = _masked_regression_loss_or_zero(loss_function_regression, matched_regression[..., :regression_dims][regression_mask], labels_regression[..., :regression_dims][regression_mask])
    element_classification_loss = F.binary_cross_entropy_with_logits(all_classification, classification_target, reduction='none')
    classification_weights = torch.full_like(element_classification_loss, eos_coef)
    classification_weights = torch.where(classification_target > 0, torch.ones_like(classification_weights), classification_weights)
    classification_loss = ((element_classification_loss * classification_weights).sum(dim=-1) / classification_weights.sum(dim=-1).clamp_min(1e-06)).mean()
    return weight * (regression_weight * regression_loss + classification_weight * classification_loss)

def calc_mse_position_regression_and_bce_classification_mask(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    epoch = kwargs.get('epoch', 1)
    model = kwargs.get('model', None)
    optimizer = kwargs.get('optimizer', None)
    validation = kwargs.get('validation', False)
    test = kwargs.get('test', False)
    weight_regression = config.WEIGHT_REGRESSION
    weight_classification = config.WEIGHT_CLASSIFICATION
    weight_precision_loss = getattr(config, 'PRECISION_LOSS_WEIGHT', 0.0)
    weight_soft_deta_loss = getattr(config, 'SOFT_DETA_LOSS_WEIGHT', 0.0)
    weight_count_over_loss = getattr(config, 'COUNT_OVER_LOSS_WEIGHT', 0.0)
    loss_function_regression = kwargs.get('loss_function_regression', nn.MSELoss())
    loss_function_classification = kwargs.get('loss_function_classification', F.binary_cross_entropy_with_logits)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    predictions_regression = predictions_regression.view(-1, config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER)
    labels_regression = labels_regression.view(-1, config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER)
    regression_loss_dims = _localization_regression_loss_dims(config, model=model, available_dims=predictions_regression.shape[-1])
    predictions_regression = predictions_regression[:, :, :regression_loss_dims]
    labels_regression = labels_regression[:, :, :regression_loss_dims]
    separation_threshold = getattr(config, 'PERSON_SEPARATION_MIN_DIST', 0.3)
    separation_weight = getattr(config, 'PERSON_SEPARATION_LOSS_WEIGHT', 10.0)
    pred_xy = predictions_regression[..., :2]
    presence_prob = torch.sigmoid(10 * predictions_classification).detach()
    pairwise_dist = torch.cdist(pred_xy.float(), pred_xy.float(), p=2)
    eye = torch.eye(pred_xy.shape[1], device=pred_xy.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = (~eye).float()
    presence_matrix = presence_prob.unsqueeze(1) * presence_prob.unsqueeze(2)
    weighted_mask = pair_mask * presence_matrix
    pair_penalty = torch.clamp(separation_threshold - pairwise_dist, min=0.0)
    weighted_penalty = pair_penalty * weighted_mask
    penalty_denom = weighted_mask.sum() + 1e-06
    base_close_pair_loss = weighted_penalty.sum() / penalty_denom
    close_pair_loss = separation_weight * base_close_pair_loss
    mask = labels_classification == 1
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions_regression)
    filtered_pred_reg = predictions_regression[mask_expanded]
    filtered_pred_cls = predictions_classification
    filtered_label_reg = labels_regression[mask_expanded]
    filtered_label_cls = labels_classification
    loss_regression = _masked_regression_loss_or_zero(loss_function_regression, filtered_pred_reg, filtered_label_reg)
    loss_classification = bce_classification_loss_with_eos_weight(filtered_pred_cls, filtered_label_cls, config, loss_function_classification)
    association_loss = output.new_zeros(())
    newborn_query_loss = _calc_motr_newborn_query_loss(output, kwargs.get('trackformer_aux'), config, reuse_cached=bool(kwargs.get('reuse_cached_auxiliary_losses', item)))
    tala_unmatched_object_loss = output.new_zeros(())
    trackformer_query_augmentation_loss = output.new_zeros(())
    slot_matching_loss = _calc_slot_head_matching_loss(output, ground_truth, config, model=model, loss_function_regression=loss_function_regression)
    temporal_presence_loss = _calc_temporal_presence_loss(output, kwargs.get('trackformer_aux'), config)
    p = torch.sigmoid(filtered_pred_cls)
    y = filtered_label_cls.float()
    soft_tp = torch.sum(p * y)
    soft_fp = torch.sum(p * (1 - y))
    soft_precision_loss = 1 - soft_tp / (soft_tp + soft_fp + 1e-06)
    soft_deta_loss = soft_deta_classification_loss(filtered_pred_cls, filtered_label_cls, fp_weight=getattr(config, 'SOFT_DETA_FP_WEIGHT', 1.0), fn_weight=getattr(config, 'SOFT_DETA_FN_WEIGHT', 1.0))
    soft_count = p.sum(dim=1)
    gt_count = y.sum(dim=1)
    overshoot = torch.clamp(soft_count - gt_count, min=0.0)
    delta = getattr(config, 'COUNT_OVER_HUBER_DELTA', 0.25)
    quadratic = 0.5 * overshoot.pow(2)
    linear = delta * overshoot - 0.5 * delta ** 2
    count_over_loss = torch.where(overshoot < delta, quadratic, linear).mean()
    if item:
        return weight_regression * loss_regression.item() + weight_classification * loss_classification.item() + weight_precision_loss * soft_precision_loss.item() + weight_soft_deta_loss * soft_deta_loss.item() + weight_count_over_loss * count_over_loss.item() + association_loss.item() + newborn_query_loss.item() + tala_unmatched_object_loss.item() + trackformer_query_augmentation_loss.item() + slot_matching_loss.item() + temporal_presence_loss.item()
    else:
        return weight_regression * loss_regression + weight_classification * loss_classification + weight_precision_loss * soft_precision_loss + weight_soft_deta_loss * soft_deta_loss + weight_count_over_loss * count_over_loss + association_loss + newborn_query_loss + tala_unmatched_object_loss + trackformer_query_augmentation_loss + slot_matching_loss + temporal_presence_loss

def calc_mse_position_regression_and_soft_f1_classification_mask(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    epoch = kwargs.get('epoch', 1)
    weight_regression = config.WEIGHT_REGRESSION
    weight_classification = config.WEIGHT_CLASSIFICATION
    loss_function_regression = kwargs.get('loss_function_regression', nn.MSELoss())
    loss_function_classification = kwargs.get('loss_function_classification', F.binary_cross_entropy_with_logits)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    (predictions_regression, labels_regression, _) = _localization_position_tensors(predictions_regression, labels_regression, config)
    mask = labels_classification == 1
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions_regression)
    filtered_pred_reg = predictions_regression[mask_expanded]
    filtered_pred_cls = predictions_classification
    filtered_label_reg = labels_regression[mask_expanded]
    filtered_label_cls = labels_classification
    loss_regression = _masked_regression_loss_or_zero(loss_function_regression, filtered_pred_reg, filtered_label_reg)
    p = torch.sigmoid(filtered_pred_cls)
    y = filtered_label_cls.float()
    sTP = torch.sum(p * y)
    sFP = torch.sum(p * (1 - y))
    sFN = torch.sum((1 - p) * y)
    prec = sTP / (sTP + sFP + 1e-06)
    rec = sTP / (sTP + sFN + 1e-06)
    soft_f1 = 2 * prec * rec / (prec + rec + 1e-06)
    loss_classification = 1 - soft_f1
    if item:
        return weight_regression * loss_regression.item() + weight_classification * loss_classification.item()
    else:
        return weight_regression * loss_regression + weight_classification * loss_classification

def calc_mse_position_regression_and_ba_classification_mask(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    epoch = kwargs.get('epoch', 1)
    weight_regression = config.WEIGHT_REGRESSION
    weight_classification = config.WEIGHT_CLASSIFICATION
    loss_function_regression = kwargs.get('loss_function_regression', nn.MSELoss())
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    (predictions_regression, labels_regression, _) = _localization_position_tensors(predictions_regression, labels_regression, config)
    mask = labels_classification == 1
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions_regression)
    filtered_pred_reg = predictions_regression[mask_expanded]
    filtered_pred_cls = predictions_classification
    filtered_label_reg = labels_regression[mask_expanded]
    filtered_label_cls = labels_classification
    loss_regression = _masked_regression_loss_or_zero(loss_function_regression, filtered_pred_reg, filtered_label_reg)
    p = torch.sigmoid(filtered_pred_cls)
    y = filtered_label_cls.float()
    sTP = torch.sum(p * y)
    sFN = torch.sum((1 - p) * y)
    sTN = torch.sum((1 - p) * (1 - y))
    sFP = torch.sum(p * (1 - y))
    sens = sTP / (sTP + sFN + 1e-06)
    spec = sTN / (sTN + sFP + 1e-06)
    soft_ba = 0.5 * (sens + spec)
    loss_classification = 1 - soft_ba
    if item:
        return weight_regression * loss_regression.item() + weight_classification * loss_classification.item()
    else:
        return weight_regression * loss_regression + weight_classification * loss_classification

def calc_mean_position_error(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    ind_loss_x = kwargs.get('ind_loss_x', False)
    (predictions_regression, predictions_classification_logits, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    predictions_classification = torch.round(nn.Sigmoid()(predictions_classification_logits))
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    (predictions_regression_position, labels_regression_position, position_dims) = _localization_position_tensors(predictions_regression, labels_regression, config)
    mask = (predictions_classification == 1) & (labels_classification == 1)
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions_regression_position)
    filtered_pred = predictions_regression_position[mask_expanded].reshape(-1, position_dims)
    filtered_label = labels_regression_position[mask_expanded].reshape(-1, position_dims)
    if filtered_pred.numel() == 0:
        return kwargs.get('undefined_value', ind_loss_x.get('MPE', False) if isinstance(ind_loss_x, dict) else False)
    distances = torch.linalg.norm(filtered_pred - filtered_label, dim=-1)
    mean_error = torch.mean(distances)
    if isinstance(ind_loss_x, dict) and 'MPE' in ind_loss_x:
        mean_error = (mean_error + ind_loss_x.get('MPE')) / 2
    return mean_error.item() if item else mean_error

def calc_median_position_error(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    ind_loss_x = kwargs.get('ind_loss_x', False)
    (predictions_regression, predictions_classification_logits, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    predictions_classification = torch.round(nn.Sigmoid()(predictions_classification_logits))
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    (predictions_regression_position, labels_regression_position, position_dims) = _localization_position_tensors(predictions_regression, labels_regression, config)
    mask = (predictions_classification == 1) & (labels_classification == 1)
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions_regression_position)
    filtered_pred = predictions_regression_position[mask_expanded].reshape(-1, position_dims)
    filtered_label = labels_regression_position[mask_expanded].reshape(-1, position_dims)
    if filtered_pred.numel() == 0:
        return kwargs.get('undefined_value', ind_loss_x.get('MedPE', False) if isinstance(ind_loss_x, dict) else False)
    distances = torch.linalg.norm(filtered_pred - filtered_label, dim=-1)
    median_error = torch.median(distances)
    if isinstance(ind_loss_x, dict) and 'MedPE' in ind_loss_x:
        median_error = (median_error + ind_loss_x.get('MedPE')) / 2
    return median_error.item() if item else median_error

def calc_mean_position_error_xy_only(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    ind_loss_x = kwargs.get('ind_loss_x', False)
    (predictions_regression, predictions_classification_logits, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    predictions_classification = torch.round(nn.Sigmoid()(predictions_classification_logits))
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    xy_dims = min(2, int(config.REGRESSION_PARAMETER))
    predictions_regression_position = predictions_regression.view(-1, config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER)[:, :, :xy_dims]
    labels_regression_position = labels_regression.view(-1, config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER)[:, :, :xy_dims]
    mask = (predictions_classification == 1) & (labels_classification == 1)
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions_regression_position)
    filtered_pred = predictions_regression_position[mask_expanded].reshape(-1, xy_dims)
    filtered_label = labels_regression_position[mask_expanded].reshape(-1, xy_dims)
    if filtered_pred.numel() == 0:
        return kwargs.get('undefined_value', ind_loss_x.get('MPEXyOnly', False) if isinstance(ind_loss_x, dict) else False)
    distances = torch.linalg.norm(filtered_pred - filtered_label, dim=-1)
    mean_error = torch.mean(distances)
    if isinstance(ind_loss_x, dict) and 'MPEXyOnly' in ind_loss_x:
        mean_error = (mean_error + ind_loss_x.get('MPEXyOnly')) / 2
    return mean_error.item() if item else mean_error

def calc_mean_velocity_error(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    ind_loss_x = kwargs.get('ind_loss_x', False)
    (predictions_regression, predictions_classification_logits, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    predictions_classification = torch.round(nn.Sigmoid()(predictions_classification_logits))
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    mask = (predictions_classification == 1) & (labels_classification == 1)
    predictions_regression = predictions_regression.view(-1, config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER)
    predictions_regression_velocity = predictions_regression[:, :, 3:]
    labels_regression = labels_regression.view(-1, config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER)
    labels_regression_velocity = labels_regression[:, :, 3:]
    velocity_dims = predictions_regression_velocity.shape[-1]
    if velocity_dims == 0:
        return kwargs.get('undefined_value', ind_loss_x.get('MVE', False) if isinstance(ind_loss_x, dict) else False)
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions_regression_velocity)
    filtered_pred = predictions_regression_velocity[mask_expanded].reshape(-1, velocity_dims)
    filtered_label = labels_regression_velocity[mask_expanded].reshape(-1, velocity_dims)
    if filtered_pred.numel() == 0:
        return kwargs.get('undefined_value', ind_loss_x.get('MVE', False) if isinstance(ind_loss_x, dict) else False)
    loss_regression_velocity = torch.mean(torch.abs(filtered_pred - filtered_label))
    if isinstance(ind_loss_x, dict) and 'MVE' in ind_loss_x:
        loss_regression_velocity = (loss_regression_velocity + ind_loss_x.get('MVE')) / 2
    return loss_regression_velocity.item() if item else loss_regression_velocity

def calc_mse_regression_loss(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    loss_function_regression = kwargs.get('loss_function_regression', nn.MSELoss())
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    configured_dims = getattr(config, 'LOCALIZATION_REGRESSION_LOSS_DIMS', None)
    if configured_dims is not None:
        (predictions_regression, labels_regression, _) = _localization_position_tensors(predictions_regression, labels_regression, config)
    loss_regression = loss_function_regression(predictions_regression, labels_regression)
    return loss_regression.item() if item else loss_regression

def calc_bce_classification_loss(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    loss_function_classification = kwargs.get('loss_function_classification', F.binary_cross_entropy_with_logits)
    loss_classification = bce_classification_loss_with_eos_weight(predictions_classification, labels_classification, config, loss_function_classification)
    return loss_classification.item() if item else loss_classification

def calc_balanced_accuracy_classification_loss(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    preds_binary = torch.round(nn.Sigmoid()(predictions_classification))
    preds_flat = preds_binary.reshape(-1)
    labels_flat = labels_classification.reshape(-1)
    TP = ((preds_flat == 1) & (labels_flat == 1)).sum().float()
    TN = ((preds_flat == 0) & (labels_flat == 0)).sum().float()
    FP = ((preds_flat == 1) & (labels_flat == 0)).sum().float()
    FN = ((preds_flat == 0) & (labels_flat == 1)).sum().float()
    ones_accuracy = TP / (TP + FN + 1e-08)
    zeros_accuracy = TN / (TN + FP + 1e-08)
    balanced_accuracy = (ones_accuracy + zeros_accuracy) / 2
    return balanced_accuracy.item() if item else balanced_accuracy

def calc_soft_balanced_accuracy_classification_loss(output, ground_truth, **kwargs):
    eps = 1e-06
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    p = torch.sigmoid(predictions_classification)
    y = labels_classification.float()
    sTP = torch.sum(p * y)
    sFN = torch.sum((1 - p) * y)
    sTN = torch.sum((1 - p) * (1 - y))
    sFP = torch.sum(p * (1 - y))
    sens = sTP / (sTP + sFN + eps)
    spec = sTN / (sTN + sFP + eps)
    soft_ba = 0.5 * (sens + spec)
    return 1 - soft_ba

def calc_accuracy_classification_loss(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    preds_binary = torch.round(nn.Sigmoid()(predictions_classification))
    preds_flat = preds_binary.reshape(-1)
    labels_flat = labels_classification.reshape(-1)
    TP = ((preds_flat == 1) & (labels_flat == 1)).sum().float()
    TN = ((preds_flat == 0) & (labels_flat == 0)).sum().float()
    FP = ((preds_flat == 1) & (labels_flat == 0)).sum().float()
    FN = ((preds_flat == 0) & (labels_flat == 1)).sum().float()
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-07)
    return accuracy.item() if item else accuracy

def calc_f1_classification_loss(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    preds_binary = torch.round(nn.Sigmoid()(predictions_classification))
    preds_flat = preds_binary.reshape(-1)
    labels_flat = labels_classification.reshape(-1)
    TP = ((preds_flat == 1) & (labels_flat == 1)).sum().float()
    TN = ((preds_flat == 0) & (labels_flat == 0)).sum().float()
    FP = ((preds_flat == 1) & (labels_flat == 0)).sum().float()
    FN = ((preds_flat == 0) & (labels_flat == 1)).sum().float()
    precision = TP / (TP + FP + 1e-07)
    recall = TP / (TP + FN + 1e-07)
    f1 = 2 * precision * recall / (precision + recall + 1e-07)
    return f1.item() if item else f1

def calc_soft_f1_loss_classification(output, ground_truth, **kwargs):
    eps = 1e-06
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    p = torch.sigmoid(predictions_classification)
    y = labels_classification.float()
    sTP = torch.sum(p * y)
    sFP = torch.sum(p * (1 - y))
    sFN = torch.sum((1 - p) * y)
    prec = sTP / (sTP + sFP + eps)
    rec = sTP / (sTP + sFN + eps)
    soft_f1 = 2 * prec * rec / (prec + rec + eps)
    return 1 - soft_f1

def calc_soft_deta_classification_loss(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    loss = soft_deta_classification_loss(predictions_classification, labels_classification, fp_weight=getattr(config, 'SOFT_DETA_FP_WEIGHT', 1.0), fn_weight=getattr(config, 'SOFT_DETA_FN_WEIGHT', 1.0))
    return loss.item() if item else loss

def calc_soft_precision_classification_loss(output, ground_truth, **kwargs):
    eps = 1e-06
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    p = torch.sigmoid(predictions_classification)
    y = labels_classification.float()
    soft_tp = torch.sum(p * y)
    soft_fp = torch.sum(p * (1 - y))
    soft_precision_loss = 1 - soft_tp / (soft_tp + soft_fp + eps)
    return soft_precision_loss.item() if item else soft_precision_loss

def calc_recall_classification_loss(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    preds_binary = torch.round(nn.Sigmoid()(predictions_classification))
    preds_flat = preds_binary.reshape(-1)
    labels_flat = labels_classification.reshape(-1)
    TP = ((preds_flat == 1) & (labels_flat == 1)).sum().float()
    TN = ((preds_flat == 0) & (labels_flat == 0)).sum().float()
    FP = ((preds_flat == 1) & (labels_flat == 0)).sum().float()
    FN = ((preds_flat == 0) & (labels_flat == 1)).sum().float()
    recall = TP / (TP + FN + 1e-07)
    return recall.item() if item else recall

def calc_precision_classification_loss(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    item = kwargs.get('item', False)
    (predictions_regression, predictions_classification, labels_regression, labels_classification) = extract_localization_components(output, ground_truth, config)
    if _use_frame_position_sort(config, kwargs):
        (predictions_regression, predictions_classification, labels_regression, labels_classification) = sort_by_position(predictions_regression, predictions_classification, labels_regression, labels_classification, config)
    preds_binary = torch.round(nn.Sigmoid()(predictions_classification))
    preds_flat = preds_binary.reshape(-1)
    labels_flat = labels_classification.reshape(-1)
    TP = ((preds_flat == 1) & (labels_flat == 1)).sum().float()
    TN = ((preds_flat == 0) & (labels_flat == 0)).sum().float()
    FP = ((preds_flat == 1) & (labels_flat == 0)).sum().float()
    FN = ((preds_flat == 0) & (labels_flat == 1)).sum().float()
    precision = TP / (TP + FP + 1e-07)
    return precision.item() if item else precision

def sort_output_by_closest_position(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    assert config is not None, 'config must be provided'
    predictions_classification = output[:, config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE:config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE + config.PNT_NUM_PEOPLE]
    rest = output[:, config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE + config.PNT_NUM_PEOPLE:]
    preds = output[:, 0:config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE].view(-1, config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER)
    (labels, _) = extract_localization_targets(ground_truth, config)
    position_dims = _localization_regression_loss_dims(config, available_dims=config.REGRESSION_PARAMETER)
    if config.PNT_NUM_PEOPLE <= 1:
        return output
    perms = torch.tensor(list(itertools.permutations(range(config.PNT_NUM_PEOPLE))), device=preds.device, dtype=torch.long)
    permuted_pred_positions = preds[:, perms, :position_dims]
    label_positions = labels[:, :, :position_dims].unsqueeze(1)
    costs = torch.norm(permuted_pred_positions - label_positions, dim=-1).sum(dim=-1)
    best_perms = perms[torch.argmin(costs, dim=1)]
    batch_indices = torch.arange(preds.shape[0], device=preds.device).unsqueeze(1)
    sorted_preds = preds[batch_indices, best_perms].reshape(-1, config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE)
    preds_cls = predictions_classification.view(-1, config.PNT_NUM_PEOPLE)
    sorted_preds_cls = preds_cls[batch_indices, best_perms]
    return torch.cat((sorted_preds, sorted_preds_cls, rest), dim=-1)
from scipy.optimize import linear_sum_assignment

def sort_output_by_closest_position_multiple_inputs(output, ground_truth, **kwargs):
    """
    Reorders variable-number predicted persons by closest (xyz) to ground-truth persons,
    and flattens back to (F, P*R + P [+ rest]).

    Behavior:
      • If exactly 2 predicted people  -> behaves like sort_output_by_closest_position()
        (explicit cost1/cost2, same tie rule <=, no Hungarian, no zero filtering)
      • Otherwise (3+ predictions)     -> uses Hungarian per frame (supports variable Np)

    Inference of number of predicted people (Np) is done from output shape.
    """
    config = kwargs.get('config')
    assert config is not None, 'config must be provided'
    R = int(config.REGRESSION_PARAMETER)
    position_dims = _localization_regression_loss_dims(config, available_dims=R)
    (F, D) = output.shape
    Np = D // (R + 1)
    leading_cols = Np * (R + 1)
    preds_reg_flat = output[:, :Np * R]
    preds_cls_flat = output[:, Np * R:Np * (R + 1)]
    rest = output[:, leading_cols:]
    preds_reg = preds_reg_flat.view(F, Np, R)
    preds_cls = preds_cls_flat.view(F, Np)
    P_gt = ground_truth.shape[1] // 6
    gt_reg = ground_truth[:, :6 * P_gt].view(F, P_gt, 6)[:, :, :R]
    gt_positions = gt_reg[:, :, :position_dims]
    if Np == 2:
        P = min(P_gt, 2)
        pr2 = preds_reg[:, :2, :]
        pc2 = preds_cls[:, :2]
        gr2 = gt_reg[:, :P, :]
        cost1 = torch.norm(pr2[:, 0, :position_dims] - gr2[:, 0, :position_dims], dim=1) + torch.norm(pr2[:, 1, :position_dims] - gr2[:, 1, :position_dims], dim=1)
        cost2 = torch.norm(pr2[:, 0, :position_dims] - gr2[:, 1, :position_dims], dim=1) + torch.norm(pr2[:, 1, :position_dims] - gr2[:, 0, :position_dims], dim=1)
        mask = cost1 <= cost2
        pr2_swapped = pr2[:, [1, 0], :]
        pc2_swapped = pc2[:, [1, 0]]
        sorted_reg = torch.where(mask.view(-1, 1, 1), pr2, pr2_swapped)
        sorted_cls = torch.where(mask.view(-1, 1), pc2, pc2_swapped)
        return torch.cat([sorted_reg.view(F, 2 * R), sorted_cls.view(F, 2), rest], dim=-1)
    P = min(P_gt, Np)
    sorted_reg = torch.zeros((F, P, R), dtype=output.dtype, device=output.device)
    sorted_cls = torch.zeros((F, P), dtype=output.dtype, device=output.device)
    if Np == 0:
        return torch.cat([sorted_reg.view(F, P * R), sorted_cls.view(F, P), rest], dim=-1)
    for t in range(F):
        pred_positions_t = preds_reg[t, :, :position_dims]
        nonzero_mask = pred_positions_t.abs().sum(dim=1) > 0
        pred_positions_use = pred_positions_t[nonzero_mask]
        idx_use = torch.nonzero(nonzero_mask, as_tuple=False).flatten()
        if pred_positions_use.shape[0] == 0:
            continue
        cost = torch.cdist(pred_positions_use.unsqueeze(0).float(), gt_positions[t].unsqueeze(0).float()).squeeze(0)
        (row_idx, col_idx) = linear_sum_assignment(cost.detach().cpu().numpy())
        chosen = [-1] * P
        for (r, c) in zip(row_idx, col_idx):
            if 0 <= c < P:
                chosen[c] = int(idx_use[r])
        for g in range(P):
            pi = chosen[g]
            if pi >= 0:
                sorted_reg[t, g, :] = preds_reg[t, pi, :]
                sorted_cls[t, g] = preds_cls[t, pi]
    return torch.cat([sorted_reg.view(F, P * R), sorted_cls.view(F, P), rest], dim=-1)

def apply_position_dependent_xy_offset(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    predictions_classification = output[:, config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE:config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE + config.PNT_NUM_PEOPLE]
    preds_regression = output[:, 0:config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE].view(-1, config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER)
    rest = output[:, config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE + config.PNT_NUM_PEOPLE:]
    ratio = preds_regression[:, :, 0] / (preds_regression[:, :, 1] + 0.001)
    delta_x = ratio / torch.sqrt(1 + ratio ** 2) * 0.12
    delta_y = 1.0 / torch.sqrt(1 + ratio ** 2) * 0.12
    mask_zero = (preds_regression[:, :, 0] == 0) & (preds_regression[:, :, 1] == 0)
    delta_x[mask_zero] = 0
    delta_y[mask_zero] = 0
    zeros_tensor = torch.zeros((preds_regression.shape[0], config.PNT_NUM_PEOPLE, config.REGRESSION_PARAMETER - 2), device=output.device)
    offsets = torch.cat((delta_x.unsqueeze(-1), delta_y.unsqueeze(-1), zeros_tensor), dim=-1)
    preds_regression = preds_regression + offsets
    preds_regression = preds_regression.view(-1, config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE)
    return torch.cat((preds_regression, predictions_classification, rest), dim=-1)

def apply_correct_classification(output, ground_truth, **kwargs):
    config = kwargs.get('config')
    preds_regression = output[:, 0:config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE]
    (_, labels_classification) = extract_localization_targets(ground_truth, config)
    rest = output[:, config.REGRESSION_PARAMETER * config.PNT_NUM_PEOPLE + config.PNT_NUM_PEOPLE:]
    return torch.cat((preds_regression, labels_classification, rest), dim=-1)
