import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import utils.train_utils
import numpy as np
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import time
from torch.utils.tensorboard import SummaryWriter
import utils.save_metrics
import signal
import sys
from contextlib import nullcontext
from itertools import permutations
from scipy.optimize import linear_sum_assignment
from hlat.utils.tracking_metrics import normalize_mean_position_error_xy, tracking_validation_score
sigint_count = 0
exiting_flag = False

def handle_sigint(signum, frame):
    global sigint_count
    global exiting_flag
    sigint_count += 1
    if sigint_count == 1:
        print('\nCaught SIGINT (Ctrl+C)! Attempting graceful shutdown... Press Ctrl+C again to force exit.')
        exiting_flag = True
    else:
        print('\nSecond Ctrl+C received. Forcing exit.')
        sys.exit(1)

class Trainer:

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config, logger: list=[None, None], save_path='./', running_local=True, gradient_clipping_max_norm=None):
        self.config = config
        self.device = self.config.DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.running_local = running_local
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.gradient_clipping_max_norm = gradient_clipping_max_norm or config.GRADIENT_CLIPPING_MAX_NORM
        self.mixed_precision_enabled = bool(getattr(config, 'MIXED_PRECISION_FLAG', False)) and self.device.type == 'cuda'
        amp_dtype = str(getattr(config, 'MIXED_PRECISION_DTYPE', 'bfloat16')).lower()
        if amp_dtype not in ('bfloat16', 'float16'):
            raise ValueError("MIXED_PRECISION_DTYPE must be either 'bfloat16' or 'float16'")
        self.mixed_precision_dtype = torch.bfloat16 if amp_dtype == 'bfloat16' else torch.float16
        self.grad_scaler = torch.amp.GradScaler('cuda', enabled=self.mixed_precision_enabled and self.mixed_precision_dtype == torch.float16) if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler(enabled=self.mixed_precision_enabled and self.mixed_precision_dtype == torch.float16)
        signal.signal(signal.SIGINT, handle_sigint)
        if logger[0] is None:
            logging.basicConfig(filename=os.path.join(self.save_path, 'logger.log'), encoding='utf-8', level=logging.INFO, format='{asctime} - {levelname} - {message}', style='{', datefmt='%Y-%m-%d %H:%M')
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger[0]
        if logger[1] is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_path, 'tensorboard_logs'))
        else:
            self.writer = logger[1]

    def _autocast_context(self):
        if not self.mixed_precision_enabled:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.mixed_precision_dtype)

    def _get_supervised_target(self, target, output=None):
        if target.dim() != 3:
            return target
        (batch_size, seq_length) = target.shape[:2]
        if output is not None and output.dim() >= 2 and (output.shape[0] == batch_size * seq_length):
            return target.reshape(batch_size * seq_length, -1)
        return target[:, -1]

    def _get_last_frame_output(self, output, target):
        if target.dim() != 3 or output.dim() < 2:
            return output
        (batch_size, seq_length) = target.shape[:2]
        if output.shape[0] != batch_size * seq_length:
            return output
        return output.reshape(batch_size, seq_length, -1)[:, -1]

    def _use_kalman_tracking_validation(self):
        return bool(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_FLAG', False))

    def _use_kalman_tracking_validation_for_best(self):
        return self._use_kalman_tracking_validation() and bool(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_SELECT_BEST_FLAG', False))

    def _should_run_kalman_tracking_validation(self, epoch):
        if not self._use_kalman_tracking_validation():
            return False
        interval = max(1, int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_INTERVAL', 1)))
        return (epoch + 1) % interval == 0

    def _validate_kalman_tracking_validation_order(self):
        if not self._use_kalman_tracking_validation():
            return
        sampler = getattr(self.val_loader, 'sampler', None)
        random_sampler_types = (torch.utils.data.RandomSampler, torch.utils.data.SubsetRandomSampler)
        if isinstance(sampler, random_sampler_types):
            raise ValueError('KALMAN_TRACKING_VALIDATION_FLAG requires validation samples in temporal order. Build the validation DataLoader with shuffle=False.')

    def _tracking_validation_score(self, metrics, id_weight, xy_weight, xy_normalization=None):
        return tracking_validation_score(metrics, selection_metric='score', id_switch_weight=id_weight, mpe_xy_weight=xy_weight, mpe_xy_normalization=xy_normalization)

    def _kalman_tracking_val_score(self, metrics):
        selection_metric = str(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_SELECTION_METRIC', 'score')).lower()
        id_weight = float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_ID_SWITCH_WEIGHT', getattr(self.config, 'MOTR_STREAMING_VALIDATION_ID_SWITCH_WEIGHT', 0.2)))
        xy_weight = float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_NMAE_XY_WEIGHT', getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MPE_XY_WEIGHT', getattr(self.config, 'MOTR_STREAMING_VALIDATION_NMAE_XY_WEIGHT', getattr(self.config, 'MOTR_STREAMING_VALIDATION_MPE_XY_WEIGHT', 0.1)))))
        xy_normalization = getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MPE_XY_NORMALIZATION', getattr(self.config, 'MOTR_STREAMING_VALIDATION_MPE_XY_NORMALIZATION', None))
        return tracking_validation_score(metrics, selection_metric=selection_metric, id_switch_weight=id_weight, mpe_xy_weight=xy_weight, mpe_xy_normalization=xy_normalization)

    def _collect_kalman_tracking_validation_batch(self, validation_outputs, output, target):
        if output.dim() != 2 or target.dim() != 2:
            return
        validation_outputs['predictions'].append(output.detach().cpu())
        validation_outputs['targets'].append(target.detach().cpu())

    def _localization_tensors_to_tracking_arrays(self, predictions, targets):
        num_people = int(getattr(self.config, 'PNT_NUM_PEOPLE', 2))
        regression_parameter = int(getattr(self.config, 'REGRESSION_PARAMETER', 3))
        pred_reg_cols = regression_parameter * num_people
        pred_cls_start = pred_reg_cols
        pred_cls_end = pred_cls_start + num_people
        target_reg_cols = 6 * num_people
        target_cls_start = 30
        target_cls_end = target_cls_start + num_people
        if predictions.shape[1] < pred_cls_end or targets.shape[1] < target_cls_end:
            raise ValueError(f'Kalman tracking validation expects localization outputs with {pred_cls_end} prediction columns and targets with {target_cls_end} columns, got {predictions.shape[1]} and {targets.shape[1]}.')
        pred_regression = torch.zeros((predictions.shape[0], num_people, 6), dtype=torch.float32)
        pred_dims = min(regression_parameter, 6)
        pred_regression[:, :, :pred_dims] = predictions[:, :pred_reg_cols].reshape(-1, num_people, regression_parameter)[:, :, :pred_dims]
        pred_scores = torch.sigmoid(predictions[:, pred_cls_start:pred_cls_end])
        pred_score_threshold = float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_PRED_SCORE_THRESH', getattr(self.config, 'KALMAN_TRACKING_VALIDATION_SCORE_THRESH', 0.5)))
        pred_active = (pred_scores >= pred_score_threshold).to(torch.float32)
        (gt_regression, gt_classification) = utils.train_utils.extract_localization_targets(targets, self.config)
        gt_regression_full = torch.zeros((targets.shape[0], num_people, 6), dtype=torch.float32, device=targets.device)
        pred_dims_for_gt = min(gt_regression.shape[-1], 6)
        gt_regression_full[:, :, :pred_dims_for_gt] = gt_regression[:, :, :pred_dims_for_gt]
        gt_regression = gt_regression_full
        gt_valid = gt_classification >= 0.5
        return (pred_regression, pred_active, gt_regression, gt_valid)

    def _validation_recording_lengths(self, total_frames):
        dataset = getattr(self.val_loader, 'dataset', None)
        lengths = getattr(dataset, 'recording_sequence_lengths', None)
        if lengths is None:
            return [total_frames]
        lengths = [int(length) for length in lengths if int(length) > 0]
        if sum(lengths) != int(total_frames):
            return [total_frames]
        return lengths

    def _run_kalman_tracking_validation_from_tensors(self, validation_outputs):
        from hlat.utils.kalman import apply_kalman
        from hlat.utils.tracking_metrics import compute_tracking_metrics_from_arrays
        if not validation_outputs['predictions']:
            return None
        predictions = torch.cat(validation_outputs['predictions'], dim=0).float()
        targets = torch.cat(validation_outputs['targets'], dim=0).float()
        (pred_regression, pred_active, gt_regression, gt_valid) = self._localization_tensors_to_tracking_arrays(predictions, targets)
        var_process_noise = float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_VAR_PROCESS_NOISE', 0.14))
        var_measurement_noise = float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_VAR_MEASUREMENT_NOISE', 3.0))
        assignment_threshold = float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_ASSIGNMENT_THRESH', 0.4))
        max_miss = int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MAX_MISS', 6))
        max_active_filters = int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MAX_ACTIVE_FILTERS', getattr(self.config, 'PNT_NUM_PEOPLE', 2)))
        filter_short_frames = int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_FILTER_SHORT_FRAMES', 0))
        kalman_dim = int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_DIM', getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MATCH_DIMS', 2)))
        kalman_regression = []
        kalman_classification = []
        kalman_track_ids = []
        next_track_id = 0
        start = 0
        for length in self._validation_recording_lengths(pred_regression.shape[0]):
            end = start + length
            (filtered_regression, filtered_classification, _, _, _, track_ids) = apply_kalman(pred_regression[start:end], pred_active[start:end], var_process_noise, var_measurement_noise, assignment_threshold, max_miss, max_active_filters, filter_short_frames, dt=1 / 15, dim=kalman_dim, return_track_ids=True)
            valid_track_ids = track_ids >= 0
            if np.any(valid_track_ids):
                track_ids[valid_track_ids] += next_track_id
                next_track_id = int(track_ids[valid_track_ids].max()) + 1
            kalman_regression.append(filtered_regression)
            kalman_classification.append(filtered_classification)
            kalman_track_ids.append(track_ids)
            start = end
        pred_regression_np = np.concatenate(kalman_regression, axis=0)
        pred_scores_np = np.concatenate(kalman_classification, axis=0)
        pred_track_ids_np = np.concatenate(kalman_track_ids, axis=0)
        sequence_breaks = []
        offset = 0
        for length in self._validation_recording_lengths(pred_regression.shape[0])[:-1]:
            offset += length
            sequence_breaks.append(offset)
        metrics = compute_tracking_metrics_from_arrays(pred_regression=pred_regression_np, pred_scores=pred_scores_np, gt_regression=gt_regression.numpy(), gt_valid=gt_valid.numpy(), score_threshold=float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_SCORE_THRESH', 0.5)), match_threshold=float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MATCH_THRESH', 1.0)), match_dims=int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MATCH_DIMS', 2)), pred_track_ids=pred_track_ids_np, scores_are_logits=False, sequence_breaks=sequence_breaks, stabilize_gt=bool(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_STABILIZE_GT', False)), gt_stabilization_threshold=getattr(self.config, 'KALMAN_TRACKING_VALIDATION_GT_STABILIZATION_THRESH', None))
        metrics['normalized_mean_position_error_xy'] = normalize_mean_position_error_xy(metrics, normalization=getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MPE_XY_NORMALIZATION', None))
        metrics['kalman_val_score'] = self._kalman_tracking_val_score(metrics)
        metrics['selection_metric'] = str(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_SELECTION_METRIC', 'score')).lower()
        metrics['evaluation_mode'] = 'kalman_tracking_validation'
        metrics['kalman_var_process_noise'] = var_process_noise
        metrics['kalman_var_measurement_noise'] = var_measurement_noise
        metrics['kalman_assignment_threshold'] = assignment_threshold
        metrics['kalman_max_miss'] = max_miss
        metrics['kalman_max_active_filters'] = max_active_filters
        metrics['kalman_dim'] = kalman_dim
        return metrics

    def _log_kalman_tracking_validation(self, metrics, epoch):
        msg = f"Epoch [{epoch + 1}/{self.config.EPOCHS}], Kalman Tracking Val Score: {metrics['kalman_val_score']:.4f}, F1: {metrics.get('f1', 0.0):.4f}, HOTA: {metrics.get('hota', 0.0):.4f}, AssA: {metrics.get('assa', 0.0):.4f}, MOTA: {(metrics.get('mota') if metrics.get('mota') is not None else 0.0):.4f}, IDSW rate: {metrics.get('id_switch_rate', 0.0) or 0.0:.4f}, MPE xy: {metrics.get('mean_position_error_xy', 0.0) or 0.0:.4f}, NMAE xy: {metrics.get('normalized_mean_position_error_xy', 1.0):.4f}"
        print(msg)
        self.logger.info(msg)
        self.writer.add_scalar('KalmanTrackingVal/Score', metrics['kalman_val_score'], epoch)
        for key in ('f1', 'hota', 'deta', 'assa', 'mota', 'precision', 'recall', 'id_switch_rate', 'mean_position_error_xy', 'normalized_mean_position_error_xy'):
            value = metrics.get(key)
            if value is not None:
                self.writer.add_scalar(f'KalmanTrackingVal/{key}', value, epoch)

    def train(self):
        start_time = time.time()
        self._validate_kalman_tracking_validation_order()
        best_val_loss = float('inf')
        lowest_val_loss = float('inf')
        best_kalman_val_score = float('-inf')
        epoch_best_val_loss = 0
        epoch_best_kalman_val_score = 0
        avg_loss_list = []
        avg_vloss_list = []
        kalman_tracking_val_history = {'kalman_val_score': [], 'f1': [], 'hota': [], 'assa': [], 'mota': [], 'precision': [], 'recall': [], 'id_switch_rate': [], 'mean_position_error_xy': [], 'normalized_mean_position_error_xy': []}
        avg_losses_lists = {}
        for loss_x in self.config.LOSSES:
            avg_losses_lists[loss_x] = []
        for ind_loss_x in self.config.IND_LOSSES:
            avg_losses_lists[ind_loss_x] = []
        avg_vlosses_lists = {}
        for loss_x in self.config.LOSSES:
            avg_vlosses_lists[loss_x] = []
        for ind_loss_x in self.config.IND_LOSSES:
            avg_vlosses_lists[ind_loss_x] = []
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.config.LR_DECAY)
        for epoch in range(self.config.EPOCHS):
            print('\n**** EPOCH %04d/%04d ****' % (epoch + 1, self.config.EPOCHS))
            self.model.train()
            running_loss = 0.0
            running_loss_values = {}
            running_vloss_values = {}
            for loss_x in self.config.LOSSES:
                running_loss_values[loss_x] = 0.0
                running_vloss_values[loss_x] = 0.0
            running_ind_loss_values = {}
            running_ind_vloss_values = {}
            running_ind_loss_counts = {}
            running_ind_vloss_counts = {}
            for ind_loss_x in self.config.IND_TEST_LOSSES:
                running_ind_loss_values[ind_loss_x] = 0.0
                running_ind_vloss_values[ind_loss_x] = 0.0
                running_ind_loss_counts[ind_loss_x] = 0
                running_ind_vloss_counts[ind_loss_x] = 0
            total_train_samples = 0
            total_val_samples = 0
            batches = iter(self.train_loader)
            if self.running_local:
                batches = tqdm(batches)
            for (step, (radar, target, *rest)) in enumerate(batches):
                self.optimizer.zero_grad(set_to_none=True)
                if rest:
                    pointcloud = rest[0]
                    pointcloud = pointcloud.to(self.device, non_blocking=True)
                radar = radar.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                with self._autocast_context():
                    if rest:
                        output = self.model((radar, pointcloud))
                    else:
                        output = self.model(radar)
                    supervised_target = self._get_supervised_target(target, output)
                    if self.config.ADAPTIONS_FLAG:
                        for adaptions in self.config.ADAPTIONS:
                            output = utils.train_utils.get_adaptions(adaptions)(output, supervised_target, config=self.config)
                    loss = self.criterion(output, supervised_target, config=self.config, epoch=epoch, model=self.model, optimizer=self.optimizer)
                running_loss += loss.item() * supervised_target.size(0)
                total_train_samples += supervised_target.size(0)
                metric_output = output.detach().float()
                for loss_x in self.config.LOSSES:
                    loss_value = utils.train_utils.get_losses(loss_x)(metric_output, supervised_target, config=self.config, item=True)
                    running_loss_values[loss_x] += loss_value * supervised_target.size(0)
                for ind_loss_x in self.config.IND_TEST_LOSSES:
                    loss_value = utils.train_utils.get_losses(ind_loss_x)(metric_output, supervised_target, config=self.config, item=True, ind_loss_x=running_ind_loss_values)
                    running_ind_loss_values[ind_loss_x] = loss_value
                self.grad_scaler.scale(loss).backward()
                if self.gradient_clipping_max_norm != None:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping_max_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            scheduler.step()
            avg_loss = running_loss / total_train_samples
            avg_loss_list.append(avg_loss)
            avg_losses = {}
            for loss_x in self.config.LOSSES:
                avg_losses[loss_x] = running_loss_values[loss_x] / total_train_samples
                avg_losses_lists[loss_x].append(avg_losses[loss_x])
            for ind_loss_x in self.config.IND_LOSSES:
                avg_losses_lists[ind_loss_x].append(running_ind_loss_values[ind_loss_x])
            print(f'Epoch [{epoch + 1}/{self.config.EPOCHS}], Train Mean Loss [cm]: {avg_loss:.2f}, ' + ' '.join((f'Train Mean {loss_x}: {avg_losses[loss_x]:.2f},' for loss_x in self.config.LOSSES)) + ', ' + ' '.join((f'Train Mean {ind_loss_x}: {running_ind_loss_values[ind_loss_x]:.2f},' for ind_loss_x in self.config.IND_LOSSES)))
            self.logger.info(f'Epoch [{epoch + 1}/{self.config.EPOCHS}], Train Mean Loss [cm]: {avg_loss:.2f}, ' + ' '.join((f'Train Mean {loss_x}: {avg_losses[loss_x]:.2f},' for loss_x in self.config.LOSSES)) + ', ' + ' '.join((f'Train Mean {ind_loss_x}: {running_ind_loss_values[ind_loss_x]:.2f},' for ind_loss_x in self.config.IND_LOSSES)))
            self.writer.add_scalar('Loss/Train', avg_loss, epoch)
            for loss_x in self.config.LOSSES:
                self.writer.add_scalar(f'{loss_x}/Train', avg_losses[loss_x], epoch)
            for ind_loss_x in self.config.IND_LOSSES:
                self.writer.add_scalar(f'{ind_loss_x}/Train', running_ind_loss_values[ind_loss_x], epoch)
            self.writer.add_scalar('Elapsed Time (seconds)', time.time() - start_time, epoch)
            self.model.eval()
            running_vloss = 0.0
            run_kalman_tracking_validation = self._should_run_kalman_tracking_validation(epoch)
            kalman_validation_outputs = {'predictions': [], 'targets': []}
            with torch.no_grad():
                for (vstep, (radar, target, *rest)) in enumerate(self.val_loader):
                    if rest:
                        pointcloud = rest[0]
                        vpointcloud = pointcloud.to(self.device, non_blocking=True)
                    vradar = radar.to(self.device, non_blocking=True)
                    validation_target = target.to(self.device, non_blocking=True)
                    with self._autocast_context():
                        if rest:
                            voutput = self.model((vradar, vpointcloud))
                        else:
                            voutput = self.model(vradar)
                        voutput = self._get_last_frame_output(voutput, validation_target)
                        vsupervised_target = self._get_supervised_target(validation_target)
                        if self.config.ADAPTIONS_FLAG:
                            for adaptions in self.config.ADAPTIONS:
                                voutput = utils.train_utils.get_adaptions(adaptions)(voutput, vsupervised_target, config=self.config)
                        val_loss = self.criterion(voutput, vsupervised_target, config=self.config, validation=True, epoch=epoch, model=self.model, optimizer=self.optimizer)
                    if run_kalman_tracking_validation:
                        self._collect_kalman_tracking_validation_batch(kalman_validation_outputs, voutput, vsupervised_target)
                    running_vloss += val_loss.item() * vsupervised_target.size(0)
                    total_val_samples += vsupervised_target.size(0)
                    metric_voutput = voutput.float()
                    for loss_x in self.config.LOSSES:
                        vloss_value = utils.train_utils.get_losses(loss_x)(metric_voutput, vsupervised_target, config=self.config)
                        running_vloss_values[loss_x] = running_vloss_values[loss_x] + vloss_value * vsupervised_target.size(0)
                    for ind_loss_x in self.config.IND_TEST_LOSSES:
                        loss_value = utils.train_utils.get_losses(ind_loss_x)(metric_voutput, vsupervised_target, config=self.config, item=True, ind_loss_x=running_ind_vloss_values)
                        running_ind_vloss_values[ind_loss_x] = loss_value
            avg_vloss = running_vloss / total_val_samples
            avg_vloss_list.append(avg_vloss)
            avg_vlosses = {}
            for loss_x in self.config.LOSSES:
                avg_vlosses[loss_x] = running_vloss_values[loss_x] / total_val_samples
                avg_vlosses_lists[loss_x].append(avg_vlosses[loss_x])
            for ind_loss_x in self.config.IND_LOSSES:
                avg_vlosses_lists[ind_loss_x].append(running_ind_vloss_values[ind_loss_x])
            print(f'Epoch [{epoch + 1}/{self.config.EPOCHS}], Val Mean Loss [cm]: {avg_vloss:.2f}, ' + ' '.join((f'Val Mean {loss_x}: {avg_vlosses[loss_x]:.2f},' for loss_x in self.config.LOSSES)) + ' '.join((f'Val Mean {ind_loss_x}: {running_ind_vloss_values[ind_loss_x]:.2f},' for ind_loss_x in self.config.IND_LOSSES)))
            self.logger.info(f'Epoch [{epoch + 1}/{self.config.EPOCHS}], Val Mean Loss [cm]: {avg_vloss:.2f}, ' + ' '.join((f'Val Mean {loss_x}: {avg_vlosses[loss_x]:.2f},' for loss_x in self.config.LOSSES)) + ' '.join((f'Val Mean {ind_loss_x}: {running_ind_vloss_values[ind_loss_x]:.2f},' for ind_loss_x in self.config.IND_LOSSES)))
            self.writer.add_scalar('Loss/Validation', avg_vloss, epoch)
            for loss_x in self.config.LOSSES:
                self.writer.add_scalar(f'{loss_x}/Validation', avg_vlosses[loss_x], epoch)
            for ind_loss_x in self.config.IND_LOSSES:
                self.writer.add_scalar(f'{ind_loss_x}/Validation', running_ind_vloss_values[ind_loss_x], epoch)
            kalman_tracking_metrics = None
            if run_kalman_tracking_validation:
                kalman_tracking_metrics = self._run_kalman_tracking_validation_from_tensors(kalman_validation_outputs)
                if kalman_tracking_metrics is not None:
                    self._log_kalman_tracking_validation(kalman_tracking_metrics, epoch)
            if self._use_kalman_tracking_validation():
                for key in kalman_tracking_val_history:
                    value = None if kalman_tracking_metrics is None else kalman_tracking_metrics.get(key)
                    kalman_tracking_val_history[key].append(value)
            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_states.pth'))
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
            if self._use_kalman_tracking_validation_for_best():
                if kalman_tracking_metrics is not None and kalman_tracking_metrics['kalman_val_score'] > best_kalman_val_score:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
                    best_kalman_val_score = kalman_tracking_metrics['kalman_val_score']
                    lowest_val_loss = avg_vloss
                    epoch_best_val_loss = epoch
                    epoch_best_kalman_val_score = epoch
                elif best_kalman_val_score == float('-inf') and avg_vloss < lowest_val_loss:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
                    lowest_val_loss = avg_vloss
                    epoch_best_val_loss = epoch
            elif avg_vloss < lowest_val_loss:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
                lowest_val_loss = avg_vloss
                epoch_best_val_loss = epoch
            if exiting_flag:
                break
        loss_dict = {'avg_train_loss': avg_loss_list, 'avg_val_loss': avg_vloss_list, 'epoch_best_val_loss': epoch_best_val_loss, 'best_val_loss': best_val_loss, 'lowest_val_loss': lowest_val_loss, 'best_model_selection': 'kalman_tracking_' + str(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_SELECTION_METRIC', 'score')).lower() if self._use_kalman_tracking_validation_for_best() else 'val_loss', 'best_kalman_val_score': None if best_kalman_val_score == float('-inf') else best_kalman_val_score, 'epoch_best_kalman_val_score': epoch_best_kalman_val_score}
        if self._use_kalman_tracking_validation():
            for (key, values) in kalman_tracking_val_history.items():
                loss_dict[f'kalman_tracking_val_{key}'] = values
        for loss_x in self.config.LOSSES:
            loss_dict[f'train_{loss_x}'] = avg_losses_lists[loss_x]
            loss_dict[f'val_{loss_x}'] = avg_vlosses_lists[loss_x]
        for ind_loss_x in self.config.IND_LOSSES:
            loss_dict[f'train_{ind_loss_x}'] = avg_losses_lists[ind_loss_x]
            loss_dict[f'val_{ind_loss_x}'] = avg_vlosses_lists[ind_loss_x]
        utils.save_metrics.save_metrics_dict_to_json(loss_dict, os.path.join(self.save_path, 'train_val_metrics.json'))
        utils.train_utils.plot_and_save_train_val_loss(avg_loss_list, avg_vloss_list, filename=self.save_path)
        return avg_vloss

class AttentionTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        signal.signal(signal.SIGINT, handle_sigint)

    def _get_trackformer_model(self):
        if callable(getattr(self.model, 'forward_frame', None)):
            return self.model
        wrapped_model = getattr(self.model, 'model', None)
        if callable(getattr(wrapped_model, 'forward_frame', None)):
            return wrapped_model
        return None

    def _supports_trackformer_api(self):
        return self._get_trackformer_model() is not None

    def _is_motr_model(self):
        trackformer_model = self._get_trackformer_model()
        if trackformer_model is None:
            return False
        return str(getattr(trackformer_model, 'tracking_approach', '')).lower() in ('motr', 'bytetrack')

    def _is_true_motr_model(self):
        trackformer_model = self._get_trackformer_model()
        if trackformer_model is None:
            return False
        return str(getattr(trackformer_model, 'tracking_approach', '')).lower() == 'motr'

    def _is_streaming_tracking_model(self):
        trackformer_model = self._get_trackformer_model()
        if trackformer_model is None:
            return False
        return str(getattr(trackformer_model, 'tracking_approach', '')).lower() in ('motr', 'trackformer', 'bytetrack')

    def _use_motr_streaming_validation(self):
        return bool(getattr(self.config, 'MOTR_STREAMING_VALIDATION_FLAG', False)) and self._is_streaming_tracking_model()

    def _use_motr_streaming_validation_for_best(self):
        return self._use_motr_streaming_validation() and bool(getattr(self.config, 'MOTR_STREAMING_VALIDATION_SELECT_BEST_FLAG', False))

    def _motr_streaming_val_score(self, metrics):
        selection_metric = str(getattr(self.config, 'MOTR_STREAMING_VALIDATION_SELECTION_METRIC', 'score')).lower()
        id_weight = float(getattr(self.config, 'MOTR_STREAMING_VALIDATION_ID_SWITCH_WEIGHT', 0.2))
        xy_weight = float(getattr(self.config, 'MOTR_STREAMING_VALIDATION_NMAE_XY_WEIGHT', getattr(self.config, 'MOTR_STREAMING_VALIDATION_MPE_XY_WEIGHT', 0.1)))
        xy_normalization = getattr(self.config, 'MOTR_STREAMING_VALIDATION_MPE_XY_NORMALIZATION', None)
        return tracking_validation_score(metrics, selection_metric=selection_metric, id_switch_weight=id_weight, mpe_xy_weight=xy_weight, mpe_xy_normalization=xy_normalization)

    def _run_motr_streaming_validation(self):
        from hlat.utils.evaluate_tba_inference import evaluate_motr_streaming, get_default_score_threshold
        motr_model = self._get_trackformer_model()
        score_threshold = getattr(self.config, 'MOTR_STREAMING_VALIDATION_SCORE_THRESH', None)
        if score_threshold is None:
            score_threshold = get_default_score_threshold(self.config)
        (metrics, _, _, _) = evaluate_motr_streaming(motr_model, self.val_loader.dataset, self.config, score_threshold=float(score_threshold), match_threshold=float(getattr(self.config, 'MOTR_STREAMING_VALIDATION_MATCH_THRESH', 1.0)), match_dims=int(getattr(self.config, 'MOTR_STREAMING_VALIDATION_MATCH_DIMS', 2)), stabilize_gt=bool(getattr(self.config, 'MOTR_STREAMING_VALIDATION_STABILIZE_GT', False)), gt_stabilization_threshold=getattr(self.config, 'MOTR_STREAMING_VALIDATION_GT_STABILIZATION_THRESH', None), save_preds=False, evaluate_warmup=bool(getattr(self.config, 'MOTR_STREAMING_VALIDATION_EVALUATE_WARMUP_FLAG', False)), context_replay=False)
        metrics['normalized_mean_position_error_xy'] = normalize_mean_position_error_xy(metrics, normalization=getattr(self.config, 'MOTR_STREAMING_VALIDATION_MPE_XY_NORMALIZATION', None))
        metrics['motr_val_score'] = self._motr_streaming_val_score(metrics)
        metrics['selection_metric'] = str(getattr(self.config, 'MOTR_STREAMING_VALIDATION_SELECTION_METRIC', 'score')).lower()
        return metrics

    def _log_motr_streaming_validation(self, metrics, epoch):
        msg = f"Epoch [{epoch + 1}/{self.config.EPOCHS}], MOTR Streaming Val Score: {metrics['motr_val_score']:.4f}, F1: {metrics.get('f1', 0.0):.4f}, HOTA: {metrics.get('hota', 0.0):.4f}, AssA: {metrics.get('assa', 0.0):.4f}, MOTA: {(metrics.get('mota') if metrics.get('mota') is not None else 0.0):.4f}, Precision: {metrics.get('precision', 0.0):.4f}, Recall: {metrics.get('recall', 0.0):.4f}, IDSW rate: {metrics.get('id_switch_rate', 0.0) or 0.0:.4f}, MPE xy: {metrics.get('mean_position_error_xy', 0.0) or 0.0:.4f}, NMAE xy: {metrics.get('normalized_mean_position_error_xy', 1.0):.4f}"
        print(msg)
        self.logger.info(msg)
        self.writer.add_scalar('MOTRStreamingVal/Score', metrics['motr_val_score'], epoch)
        for key in ('f1', 'hota', 'deta', 'assa', 'mota', 'precision', 'recall', 'id_switch_rate', 'mean_position_error_xy', 'normalized_mean_position_error_xy'):
            value = metrics.get(key)
            if value is not None:
                self.writer.add_scalar(f'MOTRStreamingVal/{key}', float(value), epoch)

    def _select_trackformer_input(self, radar, rest):
        if rest:
            return rest[0]
        return radar

    def _flatten_frame_predictions(self, regression, classification):
        regression = regression.reshape(regression.shape[0], -1)
        classification = classification.reshape(classification.shape[0], -1)
        return torch.cat((regression, classification), dim=-1)

    def _flatten_sequence_predictions(self, regression, classification):
        frame_outputs = torch.cat((regression.reshape(regression.shape[0], regression.shape[1], -1), classification.reshape(classification.shape[0], classification.shape[1], -1)), dim=-1)
        return (frame_outputs, frame_outputs.reshape(frame_outputs.shape[0] * frame_outputs.shape[1], -1))

    def _flatten_trackformer_predictions(self, regression, classification):
        if regression.dim() == 4:
            regression = regression[:, -1]
        if classification.dim() == 3:
            classification = classification[:, -1]
        return self._flatten_frame_predictions(regression, classification)

    def _forward_batch(self, radar, rest):
        if self._supports_trackformer_api():
            trackformer_input = self._select_trackformer_input(radar, rest)
            (model_output, aux) = self.model(trackformer_input, return_aux=True)
            if aux is not None:
                aux = dict(aux)
                aux['aggregated_output'] = model_output
                aux['trackformer_input'] = trackformer_input
            return (model_output, aux)
        if rest:
            return (self.model((radar, rest[0])), None)
        return (self.model(radar), None)

    def _get_supervised_output(self, model_output, trackformer_aux):
        if trackformer_aux is None:
            return model_output
        if 'pred_regression_per_frame' in trackformer_aux and 'pred_classification_per_frame' in trackformer_aux:
            final_output = self._flatten_trackformer_predictions(trackformer_aux['pred_regression_per_frame'], trackformer_aux['pred_classification_per_frame'])
            trackformer_aux['final_output'] = final_output
            return final_output
        trackformer_aux['final_output'] = model_output
        return model_output

    def _get_supervised_target(self, target, trackformer_aux):
        if target.dim() != 3:
            return target
        if trackformer_aux is not None:
            trackformer_aux['target_sequence'] = target
        return target[:, -1]

    def _split_ground_truth_frame(self, gt_frame):
        (gt_regression, gt_classification) = utils.train_utils.extract_localization_targets(gt_frame, self.config)
        return (gt_regression, gt_classification)

    def _split_ground_truth_sequence(self, target_sequence):
        (batch_size, seq_length) = target_sequence.shape[:2]
        (gt_regression, gt_classification) = self._split_ground_truth_frame(target_sequence.reshape(batch_size * seq_length, -1))
        return (gt_regression.reshape(batch_size, seq_length, gt_regression.shape[-2], gt_regression.shape[-1]), gt_classification.reshape(batch_size, seq_length, gt_classification.shape[-1]))

    def _use_track_query_sequence_matching(self):
        return bool(getattr(self.config, 'TRACK_QUERY_SEQUENCE_SLOT_MATCHING_FLAG', True))

    def _use_trackformer_query_augmentation(self, use_gt_matching):
        return bool(use_gt_matching) and bool(getattr(self.config, 'TRACKFORMER_QUERY_AUGMENTATION_FLAG', False)) and bool(getattr(self.model, 'training', False))

    def _apply_trackformer_query_false_negatives(self, track_queries, active_tracks, inactive_steps):
        drop_prob = float(getattr(self.config, 'TRACKFORMER_QUERY_FALSE_NEGATIVE_PROB', getattr(self.config, 'MOTR_QIM_RANDOM_DROP', 0.0)))
        if drop_prob <= 0.0:
            return (track_queries, active_tracks, inactive_steps, None)
        trackformer_model = self._get_trackformer_model()
        track_slots = min(track_queries.shape[1], active_tracks.shape[1])
        gt_track_slots = min(int(getattr(self.config, 'PNT_GT_NUM_PEOPLE', track_slots)), track_slots)
        drop_mask = (torch.rand(active_tracks[:, :track_slots].shape, device=active_tracks.device) < drop_prob) & active_tracks[:, :track_slots]
        if gt_track_slots < track_slots:
            drop_mask[:, gt_track_slots:] = False
        if not drop_mask.any():
            return (track_queries, active_tracks, inactive_steps, None)
        reset_queries = trackformer_model.init_track_queries(track_queries.shape[0], device=track_queries.device)
        next_queries = track_queries.clone()
        next_active = active_tracks.clone()
        next_inactive_steps = inactive_steps.clone()
        next_queries[:, :track_slots] = torch.where(drop_mask.unsqueeze(-1), reset_queries[:, :track_slots], next_queries[:, :track_slots])
        next_active[:, :track_slots] = next_active[:, :track_slots] & ~drop_mask
        next_inactive_steps[:, :track_slots] = torch.where(drop_mask, torch.zeros_like(next_inactive_steps[:, :track_slots]), next_inactive_steps[:, :track_slots])
        full_drop_mask = torch.zeros_like(active_tracks, dtype=torch.bool)
        full_drop_mask[:, :track_slots] = drop_mask
        return (next_queries, next_active, next_inactive_steps, full_drop_mask)

    def _build_trackformer_raw_query_masks(self, matched, track_classification, dropped_track_mask=None):
        assignment = matched.get('assignment')
        query_source = matched.get('query_source')
        next_active = matched.get('next_active')
        if assignment is None or query_source is None or next_active is None:
            return (None, None)
        num_track_queries = track_classification.shape[1]
        safe_assignment = assignment.clamp(min=0, max=max(num_track_queries - 1, 0))
        valid_track_assignment = (query_source == 0) & (assignment >= 0) & (assignment < num_track_queries)
        current_active = next_active[:, :assignment.shape[1]]
        positive_values = (valid_track_assignment & current_active).to(track_classification.dtype)
        positive_counts = torch.zeros_like(track_classification)
        positive_counts.scatter_add_(1, safe_assignment, positive_values)
        positive_mask = positive_counts > 0
        inactive_values = (valid_track_assignment & ~current_active).to(track_classification.dtype)
        inactive_counts = torch.zeros_like(track_classification)
        inactive_counts.scatter_add_(1, safe_assignment, inactive_values)
        inactive_matched_mask = inactive_counts > 0
        background_candidates = ~positive_mask
        false_positive_prob = float(getattr(self.config, 'TRACKFORMER_QUERY_FALSE_POSITIVE_PROB', getattr(self.config, 'MOTR_QIM_FP_RATIO', 0.0)))
        if false_positive_prob >= 1.0:
            background_mask = background_candidates.clone()
        elif false_positive_prob > 0.0:
            background_mask = (torch.rand(track_classification.shape, device=track_classification.device) < false_positive_prob) & background_candidates
        else:
            background_mask = torch.zeros_like(background_candidates)
        background_mask = background_mask | inactive_matched_mask & background_candidates
        if dropped_track_mask is not None:
            background_mask = background_mask | dropped_track_mask[:, :num_track_queries] & background_candidates
        return (positive_mask, background_mask)

    def _match_sequence_slots(self, pred_regression, pred_classification, target_sequence):
        if not self._use_track_query_sequence_matching():
            return (pred_regression, pred_classification, None, None)
        if pred_classification.dim() == pred_regression.dim():
            pred_classification = pred_classification.squeeze(-1)
        (gt_regression, gt_classification) = self._split_ground_truth_sequence(target_sequence)
        (batch_size, seq_length, pred_slots) = pred_classification.shape
        gt_slots = gt_classification.shape[-1]
        if pred_slots != gt_slots:
            return (pred_regression, pred_classification, None, None)
        dims = min(int(getattr(self.config, 'TRACK_QUERY_SEQUENCE_MATCH_DIMS', 3)), pred_regression.shape[-1], gt_regression.shape[-1])
        pos_weight = float(getattr(self.config, 'TRACK_QUERY_SEQUENCE_MATCH_COST_POS', 1.0))
        cls_weight = float(getattr(self.config, 'TRACK_QUERY_SEQUENCE_MATCH_COST_CLS', getattr(self.config, 'MATCH_COST_CLS', 0.1)))
        with torch.no_grad():
            pred_pos = pred_regression[..., :dims].float()
            gt_pos = gt_regression[..., :dims].float()
            gt_active = gt_classification.float()
            active_mask = gt_active > 0.5
            position_distance = torch.linalg.vector_norm(pred_pos.unsqueeze(3) - gt_pos.unsqueeze(2), dim=-1)
            active_weights = active_mask.unsqueeze(2).to(position_distance.dtype)
            active_counts = active_weights.sum(dim=1).clamp_min(1.0)
            position_cost = (position_distance * active_weights).sum(dim=1) / active_counts
            classification_cost = F.binary_cross_entropy_with_logits(pred_classification.float().unsqueeze(3).expand(-1, -1, -1, gt_slots), gt_active.unsqueeze(2).expand(-1, -1, pred_slots, -1), reduction='none').mean(dim=1)
            assignment_costs = pos_weight * position_cost + cls_weight * classification_cost
            permutation_cache = getattr(self, '_slot_permutation_cache', {})
            cache_key = (pred_slots, pred_regression.device)
            candidate_permutations = permutation_cache.get(cache_key)
            if candidate_permutations is None:
                candidate_permutations = torch.tensor(tuple(permutations(range(pred_slots))), device=pred_regression.device, dtype=torch.long)
                permutation_cache[cache_key] = candidate_permutations
                self._slot_permutation_cache = permutation_cache
            costs_by_gt_slot = assignment_costs.transpose(1, 2)
            permutation_costs = torch.gather(costs_by_gt_slot.unsqueeze(1).expand(-1, candidate_permutations.shape[0], -1, -1), dim=3, index=candidate_permutations.view(1, candidate_permutations.shape[0], gt_slots, 1).expand(batch_size, -1, -1, -1)).squeeze(3).sum(dim=2)
            best_permutation = permutation_costs.argmin(dim=1)
            slot_indices = candidate_permutations[best_permutation]
        regression_indices = slot_indices[:, None, :, None].expand(-1, seq_length, -1, pred_regression.shape[-1])
        classification_indices = slot_indices[:, None, :].expand(-1, seq_length, -1)
        matched_regression = torch.gather(pred_regression, dim=2, index=regression_indices)
        matched_classification = torch.gather(pred_classification, dim=2, index=classification_indices)
        return (matched_regression, matched_classification, slot_indices, assignment_costs)

    def _match_all_queries_to_gt(self, track_regression, track_classification, object_regression, object_classification, gt_regression, gt_classification, prev_active):
        active_gt_slots = torch.nonzero(gt_classification > 0.5, as_tuple=False).flatten()
        if active_gt_slots.numel() == 0:
            return []
        candidate_regressions = []
        candidate_classifications = []
        candidate_sources = []
        candidate_indices = []
        active_track_slots = torch.nonzero(prev_active, as_tuple=False).flatten()
        active_track_slots = active_track_slots[active_track_slots < track_regression.shape[0]]
        if active_track_slots.numel() > 0:
            candidate_regressions.append(track_regression[active_track_slots])
            candidate_classifications.append(track_classification[active_track_slots])
            candidate_sources.extend([0] * active_track_slots.numel())
            candidate_indices.extend(active_track_slots.tolist())
        if object_regression is not None and object_classification is not None:
            candidate_regressions.append(object_regression)
            candidate_classifications.append(object_classification)
            candidate_sources.extend([1] * object_regression.shape[0])
            candidate_indices.extend(range(object_regression.shape[0]))
        if not candidate_regressions:
            fallback_slots = torch.arange(min(track_regression.shape[0], gt_regression.shape[0]), device=track_regression.device)
            candidate_regressions.append(track_regression[fallback_slots])
            candidate_classifications.append(track_classification[fallback_slots])
            candidate_sources.extend([0] * fallback_slots.numel())
            candidate_indices.extend(fallback_slots.tolist())
        query_regression = torch.cat(candidate_regressions, dim=0)
        query_classification = torch.cat(candidate_classifications, dim=0)
        gt_slots = active_gt_slots.tolist()
        gt_active_regression = gt_regression[active_gt_slots]
        pos_dim = min(3, query_regression.shape[-1], gt_active_regression.shape[-1])
        pos_cost = torch.cdist(query_regression[:, :pos_dim].unsqueeze(0).float(), gt_active_regression[:, :pos_dim].unsqueeze(0).float()).squeeze(0)
        pred_prob = torch.sigmoid(query_classification).clamp_min(1e-06)
        cls_cost = (-torch.log(pred_prob)).unsqueeze(1).expand(-1, active_gt_slots.numel())
        source_cost = torch.zeros_like(pos_cost)
        object_existing_penalty = getattr(self.config, 'MATCH_COST_OBJECT_TO_EXISTING', 0.2)
        if object_existing_penalty:
            source_tensor = torch.tensor(candidate_sources, device=pos_cost.device, dtype=torch.long)
            previously_active_gt = prev_active[active_gt_slots].float()
            source_cost = (source_tensor == 1).float().unsqueeze(1) * previously_active_gt.unsqueeze(0) * object_existing_penalty
        match_cost_pos = getattr(self.config, 'MATCH_COST_POS', 1.0)
        match_cost_cls = getattr(self.config, 'MATCH_COST_CLS', 0.5)
        cost = match_cost_pos * pos_cost + match_cost_cls * cls_cost + source_cost
        (row_idx, col_idx) = linear_sum_assignment(cost.detach().cpu().numpy())
        matches = []
        for (row, col) in zip(row_idx, col_idx):
            if col >= len(gt_slots):
                continue
            matches.append({'gt_slot': gt_slots[col], 'source': candidate_sources[row], 'query_idx': int(candidate_indices[row])})
        return matches

    def _match_and_propagate_queries(self, frame_out, gt_frame, prev_track_queries, prev_active):
        """
        Build identity-constrained TrackFormer training targets.

        A GT identity that was active in the previous frame stays bound to its
        track-query slot. Only newly appearing identities are matched to object
        queries. This mirrors TrackFormer's track-query assignment instead of
        rematching persistent identities against all queries every frame.
        """
        (gt_regression, gt_classification) = self._split_ground_truth_frame(gt_frame)
        track_queries = frame_out['track_queries']
        object_queries = frame_out.get('object_queries')
        track_regression = frame_out.get('pred_regression_track')
        if track_regression is None:
            track_regression = frame_out['pred_regression']
        track_classification = frame_out.get('pred_classification_track')
        if track_classification is None:
            track_classification = frame_out['pred_classification']
        object_regression = frame_out.get('pred_regression_object')
        object_classification = frame_out.get('pred_classification_object')
        batch_size = track_queries.shape[0]
        num_people = gt_regression.shape[1]
        num_track_queries = track_queries.shape[1]
        reg_dim = track_regression.shape[-1]
        track_slots = min(num_people, num_track_queries)
        next_track_queries = track_queries.clone()
        reordered_regression = torch.zeros((batch_size, num_people, reg_dim), device=track_regression.device, dtype=track_regression.dtype)
        reordered_classification = torch.full((batch_size, num_people), -10.0, device=track_classification.device, dtype=track_classification.dtype)
        assignment = torch.full((batch_size, num_people), -1, device=track_queries.device, dtype=torch.long)
        query_source = torch.full((batch_size, num_people), -1, device=track_queries.device, dtype=torch.long)
        next_active = gt_classification > 0.5
        false_positive_track_mask = torch.zeros((batch_size, num_people), device=track_queries.device, dtype=torch.bool)
        false_positive_object_indices = torch.full((batch_size, num_people), -1, device=track_queries.device, dtype=torch.long)
        for batch_idx in range(batch_size):
            current_active = gt_classification[batch_idx] > 0.5
            previous_active = prev_active[batch_idx, :num_people]
            used_object_indices = set()
            for gt_slot in range(track_slots):
                if not bool(previous_active[gt_slot]):
                    continue
                reordered_regression[batch_idx, gt_slot] = track_regression[batch_idx, gt_slot]
                reordered_classification[batch_idx, gt_slot] = track_classification[batch_idx, gt_slot]
                assignment[batch_idx, gt_slot] = gt_slot
                query_source[batch_idx, gt_slot] = 0
                next_track_queries[batch_idx, gt_slot] = track_queries[batch_idx, gt_slot]
            newborn_slots = torch.nonzero(current_active & ~previous_active, as_tuple=False).flatten()
            if newborn_slots.numel() > 0 and object_regression is not None and (object_classification is not None) and (object_queries is not None):
                pos_dim = min(int(getattr(self.config, 'MOTR_NEWBORN_MATCH_DIMS', 2)), object_regression.shape[-1], gt_regression.shape[-1])
                position_cost = torch.cdist(object_regression[batch_idx, :, :pos_dim].unsqueeze(0).float(), gt_regression[batch_idx, newborn_slots, :pos_dim].unsqueeze(0).float()).squeeze(0)
                classification_cost = -F.logsigmoid(object_classification[batch_idx].float()).unsqueeze(1)
                cost = float(getattr(self.config, 'MOTR_NEWBORN_MATCH_COST_POS', getattr(self.config, 'MATCH_COST_POS', 1.0))) * position_cost + float(getattr(self.config, 'MOTR_NEWBORN_MATCH_COST_CLS', getattr(self.config, 'MATCH_COST_CLS', 0.1))) * classification_cost
                (object_rows, newborn_cols) = linear_sum_assignment(cost.detach().cpu().numpy())
                for (object_idx, newborn_col) in zip(object_rows, newborn_cols):
                    gt_slot = int(newborn_slots[newborn_col].item())
                    object_idx = int(object_idx)
                    reordered_regression[batch_idx, gt_slot] = object_regression[batch_idx, object_idx]
                    reordered_classification[batch_idx, gt_slot] = object_classification[batch_idx, object_idx]
                    assignment[batch_idx, gt_slot] = object_idx
                    query_source[batch_idx, gt_slot] = 1
                    used_object_indices.add(object_idx)
                    if gt_slot < num_track_queries:
                        next_track_queries[batch_idx, gt_slot] = object_queries[batch_idx, object_idx]
            if self._use_trackformer_query_augmentation(use_gt_matching=True) and object_queries is not None and (object_classification is not None):
                false_positive_prob = float(getattr(self.config, 'TRACKFORMER_QUERY_FALSE_POSITIVE_PROB', getattr(self.config, 'MOTR_QIM_FP_RATIO', 0.0)))
                if false_positive_prob > 0.0:
                    gt_track_slots = min(int(getattr(self.config, 'PNT_GT_NUM_PEOPLE', num_people)), track_slots)
                    padded_track_slots = torch.arange(gt_track_slots, track_slots, device=track_queries.device)
                    free_slots = padded_track_slots[~next_active[batch_idx, padded_track_slots]]
                    if free_slots.numel() > 0:
                        object_scores = torch.sigmoid(object_classification[batch_idx])
                        candidate_objects = [object_idx for object_idx in torch.argsort(object_scores, descending=True).tolist() if object_idx not in used_object_indices]
                        sampled_objects = [object_idx for object_idx in candidate_objects if float(torch.rand((), device=object_scores.device).item()) < false_positive_prob]
                        for (slot_tensor, object_idx) in zip(free_slots, sampled_objects):
                            slot = int(slot_tensor.item())
                            next_track_queries[batch_idx, slot] = object_queries[batch_idx, object_idx]
                            next_active[batch_idx, slot] = True
                            false_positive_track_mask[batch_idx, slot] = True
                            false_positive_object_indices[batch_idx, slot] = int(object_idx)
        return {'next_track_queries': next_track_queries, 'matched_regression': reordered_regression, 'matched_classification': reordered_classification, 'assignment': assignment, 'query_source': query_source, 'next_active': next_active, 'false_positive_track_mask': false_positive_track_mask, 'false_positive_object_indices': false_positive_object_indices}

    def _propagate_queries_without_gt(self, frame_out, prev_active, prev_inactive_steps=None):
        trackformer_model = self._get_trackformer_model()
        propagate = getattr(trackformer_model, '_propagate_without_gt', None)
        if callable(propagate):
            (next_track_queries, next_active, output_regression, output_classification) = propagate(frame_out, prev_active, prev_inactive_steps)
            return {'next_track_queries': next_track_queries, 'next_active': next_active, 'next_inactive_steps': frame_out.get('next_inactive_steps'), 'matched_regression': output_regression, 'matched_classification': output_classification}
        track_queries = frame_out['track_queries']
        object_queries = frame_out.get('object_queries')
        track_regression = frame_out.get('pred_regression_track')
        if track_regression is None:
            track_regression = frame_out['pred_regression']
        track_classification = frame_out.get('pred_classification_track')
        if track_classification is None:
            track_classification = frame_out['pred_classification']
        object_regression = frame_out.get('pred_regression_object')
        object_classification = frame_out.get('pred_classification_object')
        batch_size = track_queries.shape[0]
        num_people = self.config.PNT_NUM_PEOPLE
        num_track_queries = track_queries.shape[1]
        track_slots = min(num_people, num_track_queries)
        reg_dim = track_regression.shape[-1]
        keep_thresh = getattr(self.config, 'TRACKFORMER_KEEP_TRACK_SCORE_THRESH', 0.35)
        new_thresh = getattr(self.config, 'TRACKFORMER_NEW_TRACK_SCORE_THRESH', 0.5)
        next_track_queries = track_queries.clone()
        next_active = prev_active.clone()
        output_regression = torch.zeros((batch_size, num_people, reg_dim), device=track_regression.device, dtype=track_regression.dtype)
        output_classification = torch.full((batch_size, num_people), -10.0, device=track_classification.device, dtype=track_classification.dtype)
        for slot in range(track_slots):
            output_regression[:, slot] = track_regression[:, slot]
            output_classification[:, slot] = track_classification[:, slot]
            keep_mask = torch.sigmoid(track_classification[:, slot]) >= keep_thresh
            next_active[:, slot] = prev_active[:, slot] & keep_mask
        if object_regression is None or object_classification is None or object_queries is None:
            return {'next_track_queries': next_track_queries, 'next_active': next_active, 'matched_regression': output_regression, 'matched_classification': output_classification}
        object_scores = torch.sigmoid(object_classification)
        candidate_order = torch.argsort(object_scores, dim=1, descending=True)
        sorted_scores = torch.gather(object_scores, 1, candidate_order)
        free_mask = ~next_active[:, :track_slots]
        free_rank = free_mask.to(torch.long).cumsum(dim=1) - 1
        safe_rank = free_rank.clamp(min=0, max=candidate_order.shape[1] - 1)
        assigned_objects = torch.gather(candidate_order, 1, safe_rank)
        assigned_scores = torch.gather(sorted_scores, 1, safe_rank)
        promote_mask = free_mask & (free_rank >= 0) & (free_rank < candidate_order.shape[1]) & (assigned_scores >= new_thresh)
        query_indices = assigned_objects.unsqueeze(-1).expand(-1, -1, object_queries.shape[-1])
        regression_indices = assigned_objects.unsqueeze(-1).expand(-1, -1, object_regression.shape[-1])
        promoted_queries = torch.gather(object_queries, 1, query_indices)
        promoted_regression = torch.gather(object_regression, 1, regression_indices)
        promoted_classification = torch.gather(object_classification, 1, assigned_objects)
        next_track_queries[:, :track_slots] = torch.where(promote_mask.unsqueeze(-1), promoted_queries, next_track_queries[:, :track_slots])
        next_active[:, :track_slots] = next_active[:, :track_slots] | promote_mask
        output_regression[:, :track_slots] = torch.where(promote_mask.unsqueeze(-1), promoted_regression, output_regression[:, :track_slots])
        output_classification[:, :track_slots] = torch.where(promote_mask, promoted_classification, output_classification[:, :track_slots])
        return {'next_track_queries': next_track_queries, 'next_active': next_active, 'matched_regression': output_regression, 'matched_classification': output_classification}

    def _forward_trackformer_sequence(self, trackformer_input, target_sequence, use_gt_matching=True, apply_sequence_slot_matching=True):
        trackformer_model = self._get_trackformer_model()
        if str(getattr(trackformer_model, 'training_assignment', '')).lower() == 'recursive_tala':
            return trackformer_model.forward_train_sequence(trackformer_input, target_sequence)
        if self._is_motr_model():
            (_, aux) = trackformer_model(trackformer_input, return_aux=True)
            raw_pred_regression = aux['pred_regression_per_frame']
            raw_pred_classification = aux['pred_classification_per_frame']
            if apply_sequence_slot_matching:
                (pred_regression, pred_classification, sequence_slot_indices, sequence_match_cost) = self._match_sequence_slots(raw_pred_regression, raw_pred_classification, target_sequence)
            else:
                pred_regression = raw_pred_regression
                pred_classification = raw_pred_classification
                sequence_slot_indices = None
                sequence_match_cost = None
            (frame_outputs, flattened_outputs) = self._flatten_sequence_predictions(pred_regression, pred_classification)
            flattened_targets = target_sequence.reshape(target_sequence.shape[0] * target_sequence.shape[1], -1)
            aux = dict(aux)
            aux.update({'target_sequence': target_sequence, 'trackformer_input': trackformer_input, 'frame_outputs': frame_outputs, 'final_output': frame_outputs[:, -1], 'sequence_supervised_output': flattened_outputs, 'sequence_supervised_target': flattened_targets, 'state_pred_regression_per_frame': aux['pred_regression_per_frame'], 'state_pred_classification_per_frame': aux['pred_classification_per_frame'], 'pred_regression_per_frame': pred_regression, 'pred_classification_per_frame': pred_classification})
            if sequence_slot_indices is not None:
                aux['sequence_slot_matching_applied'] = True
                aux['sequence_slot_indices'] = sequence_slot_indices
                aux['sequence_slot_match_cost'] = sequence_match_cost
            return (flattened_outputs, flattened_targets, aux)
        encoded_frames = trackformer_model.encode_sequence(trackformer_input)
        (batch_size, seq_length) = encoded_frames.shape[:2]
        initial_queries = trackformer_model.init_track_queries(batch_size, device=encoded_frames.device)
        track_queries = initial_queries
        active_tracks = torch.zeros(batch_size, int(getattr(trackformer_model, 'num_track_queries', self.config.PNT_NUM_PEOPLE)), device=encoded_frames.device, dtype=torch.bool)
        inactive_steps = torch.zeros_like(active_tracks, dtype=torch.long)
        frame_outputs = []
        raw_query_history = []
        raw_track_classification_history = []
        propagated_query_history = []
        matched_regression_history = []
        matched_classification_history = []
        object_regression_history = []
        object_classification_history = []
        assignment_history = []
        query_source_history = []
        raw_track_positive_mask_history = []
        raw_track_background_mask_history = []
        dropped_track_mask_history = []
        injected_false_positive_mask_history = []
        injected_false_positive_object_history = []
        for t in range(seq_length):
            dropped_track_mask = None
            if self._use_trackformer_query_augmentation(use_gt_matching):
                (track_queries, active_tracks, inactive_steps, dropped_track_mask) = self._apply_trackformer_query_false_negatives(track_queries, active_tracks, inactive_steps)
            frame_out = trackformer_model.forward_frame(encoded_frames[:, t], track_queries=track_queries, frame_already_encoded=True)
            raw_query_history.append(frame_out['track_queries'])
            raw_track_classification = frame_out.get('pred_classification_track')
            if raw_track_classification is None:
                raw_track_classification = frame_out['pred_classification']
            raw_track_classification_history.append(raw_track_classification)
            if frame_out.get('pred_regression_object') is not None and frame_out.get('pred_classification_object') is not None:
                object_regression_history.append(frame_out['pred_regression_object'])
                object_classification_history.append(frame_out['pred_classification_object'])
            if use_gt_matching:
                matched = self._match_and_propagate_queries(frame_out=frame_out, gt_frame=target_sequence[:, t], prev_track_queries=track_queries, prev_active=active_tracks)
                flat_output_t = self._flatten_frame_predictions(matched['matched_regression'], matched['matched_classification'])
                frame_outputs.append(flat_output_t)
                propagated_query_history.append(matched['next_track_queries'])
                matched_regression_history.append(matched['matched_regression'])
                matched_classification_history.append(matched['matched_classification'])
                assignment_history.append(matched['assignment'])
                query_source_history.append(matched['query_source'])
                injected_false_positive_mask_history.append(matched['false_positive_track_mask'])
                injected_false_positive_object_history.append(matched['false_positive_object_indices'])
                if self._use_trackformer_query_augmentation(use_gt_matching):
                    (positive_mask, background_mask) = self._build_trackformer_raw_query_masks(matched, raw_track_classification, dropped_track_mask=dropped_track_mask)
                    if positive_mask is not None and background_mask is not None:
                        raw_track_positive_mask_history.append(positive_mask)
                        raw_track_background_mask_history.append(background_mask)
                        if dropped_track_mask is None:
                            dropped_track_mask = torch.zeros_like(active_tracks, dtype=torch.bool)
                        dropped_track_mask_history.append(dropped_track_mask)
                track_queries = matched['next_track_queries']
                active_tracks = matched['next_active']
            else:
                propagated = self._propagate_queries_without_gt(frame_out=frame_out, prev_active=active_tracks, prev_inactive_steps=inactive_steps)
                flat_output_t = self._flatten_frame_predictions(propagated['matched_regression'], propagated['matched_classification'])
                frame_outputs.append(flat_output_t)
                propagated_query_history.append(propagated['next_track_queries'])
                matched_regression_history.append(propagated['matched_regression'])
                matched_classification_history.append(propagated['matched_classification'])
                track_queries = propagated['next_track_queries']
                active_tracks = propagated['next_active']
                if propagated.get('next_inactive_steps') is not None:
                    inactive_steps = propagated['next_inactive_steps']
        frame_outputs = torch.stack(frame_outputs, dim=1)
        raw_query_history = torch.stack(raw_query_history, dim=1)
        raw_track_classification_history = torch.stack(raw_track_classification_history, dim=1)
        propagated_query_history = torch.stack(propagated_query_history, dim=1)
        matched_regression_history = torch.stack(matched_regression_history, dim=1)
        matched_classification_history = torch.stack(matched_classification_history, dim=1)
        if len(object_regression_history) == seq_length:
            object_regression_history = torch.stack(object_regression_history, dim=1)
            object_classification_history = torch.stack(object_classification_history, dim=1)
        else:
            object_regression_history = None
            object_classification_history = None
        if use_gt_matching:
            assignment_history = torch.stack(assignment_history, dim=1)
            query_source_history = torch.stack(query_source_history, dim=1)
            injected_false_positive_mask_history = torch.stack(injected_false_positive_mask_history, dim=1)
            injected_false_positive_object_history = torch.stack(injected_false_positive_object_history, dim=1)
        else:
            assignment_history = None
            query_source_history = None
            injected_false_positive_mask_history = None
            injected_false_positive_object_history = None
        if len(raw_track_background_mask_history) == seq_length:
            raw_track_positive_mask_history = torch.stack(raw_track_positive_mask_history, dim=1)
            raw_track_background_mask_history = torch.stack(raw_track_background_mask_history, dim=1)
            dropped_track_mask_history = torch.stack(dropped_track_mask_history, dim=1)
        else:
            raw_track_positive_mask_history = None
            raw_track_background_mask_history = None
            dropped_track_mask_history = None
        if use_gt_matching:
            sequence_slot_indices = None
            sequence_match_cost = None
        elif apply_sequence_slot_matching:
            (matched_regression_history, matched_classification_history, sequence_slot_indices, sequence_match_cost) = self._match_sequence_slots(matched_regression_history, matched_classification_history, target_sequence)
        else:
            sequence_slot_indices = None
            sequence_match_cost = None
        (frame_outputs, flattened_outputs) = self._flatten_sequence_predictions(matched_regression_history, matched_classification_history)
        flattened_targets = target_sequence.reshape(batch_size * seq_length, -1)
        aux = {'encoded_frames': encoded_frames, 'target_sequence': target_sequence, 'trackformer_input': trackformer_input, 'frame_outputs': frame_outputs, 'track_queries_per_frame': raw_query_history, 'raw_track_classification_per_frame': raw_track_classification_history, 'propagated_track_queries_per_frame': propagated_query_history, 'pred_regression_per_frame': matched_regression_history, 'pred_classification_per_frame': matched_classification_history, 'final_track_queries': track_queries, 'final_output': frame_outputs[:, -1], 'sequence_supervised_output': flattened_outputs, 'sequence_supervised_target': flattened_targets}
        if use_gt_matching:
            aux['matched_query_indices'] = assignment_history
            aux['matched_query_source'] = query_source_history
            aux['trackformer_injected_false_positive_mask_per_frame'] = injected_false_positive_mask_history
            aux['trackformer_injected_false_positive_object_per_frame'] = injected_false_positive_object_history
            aux['gt_propagation_matching_applied'] = True
            if raw_track_background_mask_history is not None:
                aux['trackformer_raw_track_positive_mask_per_frame'] = raw_track_positive_mask_history
                aux['trackformer_raw_track_background_mask_per_frame'] = raw_track_background_mask_history
                aux['trackformer_dropped_track_mask_per_frame'] = dropped_track_mask_history
        if object_regression_history is not None:
            aux['pred_regression_object_per_frame'] = object_regression_history
            aux['pred_classification_object_per_frame'] = object_classification_history
        if sequence_slot_indices is not None:
            aux['sequence_slot_matching_applied'] = True
            aux['sequence_slot_indices'] = sequence_slot_indices
            aux['sequence_slot_match_cost'] = sequence_match_cost
        return (flattened_outputs, flattened_targets, aux)

    def train(self):
        start_time = time.time()
        best_val_loss = float('inf')
        lowest_val_loss = float('inf')
        best_motr_val_score = float('-inf')
        epoch_best_val_loss = 0
        epoch_best_motr_val_score = 0
        avg_loss_list = []
        avg_vloss_list = []
        motr_streaming_val_history = {'motr_val_score': [], 'f1': [], 'hota': [], 'assa': [], 'mota': [], 'precision': [], 'recall': [], 'id_switch_rate': [], 'mean_position_error_xy': [], 'normalized_mean_position_error_xy': []}
        avg_losses_lists = {}
        for loss_x in self.config.LOSSES:
            avg_losses_lists[loss_x] = []
        for ind_loss_x in self.config.IND_LOSSES:
            avg_losses_lists[ind_loss_x] = []
        avg_vlosses_lists = {}
        for loss_x in self.config.LOSSES:
            avg_vlosses_lists[loss_x] = []
        for ind_loss_x in self.config.IND_LOSSES:
            avg_vlosses_lists[ind_loss_x] = []
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.config.LR_DECAY)
        for epoch in range(self.config.EPOCHS):
            print('\n**** EPOCH %04d/%04d ****' % (epoch + 1, self.config.EPOCHS))
            self.model.train()
            running_loss = 0.0
            running_loss_values = {}
            running_vloss_values = {}
            for loss_x in self.config.LOSSES:
                running_loss_values[loss_x] = 0.0
                running_vloss_values[loss_x] = 0.0
            running_ind_loss_values = {}
            running_ind_vloss_values = {}
            running_ind_loss_counts = {}
            running_ind_vloss_counts = {}
            for ind_loss_x in self.config.IND_TEST_LOSSES:
                running_ind_loss_values[ind_loss_x] = 0.0
                running_ind_vloss_values[ind_loss_x] = 0.0
                running_ind_loss_counts[ind_loss_x] = 0
                running_ind_vloss_counts[ind_loss_x] = 0
            total_train_samples = 0
            total_val_samples = 0
            batches = iter(self.train_loader)
            if self.running_local:
                batches = tqdm(batches)
            for (step, (radar, target, *rest)) in enumerate(batches):
                self.optimizer.zero_grad(set_to_none=True)
                if rest:
                    pointcloud = rest[0]
                    pointcloud = pointcloud.to(self.device, non_blocking=True)
                radar = radar.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                with self._autocast_context():
                    if self._supports_trackformer_api() and target.dim() == 3:
                        trackformer_input = self._select_trackformer_input(radar, rest)
                        (output, supervised_target, trackformer_aux) = self._forward_trackformer_sequence(trackformer_input, target, use_gt_matching=bool(getattr(self.config, 'TRACKFORMER_GT_PROPAGATION_FLAG', False)))
                    else:
                        (model_output, trackformer_aux) = self._forward_batch(radar, rest)
                        output = self._get_supervised_output(model_output, trackformer_aux)
                        supervised_target = self._get_supervised_target(target, trackformer_aux)
                    if self.config.ADAPTIONS_FLAG:
                        for adaptions in self.config.ADAPTIONS:
                            output = utils.train_utils.get_adaptions(adaptions)(output, supervised_target, config=self.config)
                        if trackformer_aux is not None:
                            trackformer_aux['adapted_final_output'] = output
                    loss = self.criterion(output, supervised_target, config=self.config, epoch=epoch, model=self.model, optimizer=self.optimizer, trackformer_aux=trackformer_aux)
                running_loss += loss.item() * supervised_target.size(0)
                total_train_samples += supervised_target.size(0)
                metric_output = output.detach().float()
                for loss_x in self.config.LOSSES:
                    loss_value = utils.train_utils.get_losses(loss_x)(metric_output, supervised_target, config=self.config, item=True, trackformer_aux=trackformer_aux)
                    running_loss_values[loss_x] += loss_value * supervised_target.size(0)
                for ind_loss_x in self.config.IND_TEST_LOSSES:
                    loss_value = utils.train_utils.get_losses(ind_loss_x)(metric_output, supervised_target, config=self.config, item=True, undefined_value=float('nan'), trackformer_aux=trackformer_aux)
                    loss_value = loss_value.item() if hasattr(loss_value, 'item') else loss_value
                    loss_value = float(loss_value)
                    if np.isfinite(loss_value):
                        running_ind_loss_values[ind_loss_x] += loss_value * supervised_target.size(0)
                        running_ind_loss_counts[ind_loss_x] += supervised_target.size(0)
                self.grad_scaler.scale(loss).backward()
                if self.gradient_clipping_max_norm != None:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping_max_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            scheduler.step()
            avg_loss = running_loss / total_train_samples
            avg_loss_list.append(avg_loss)
            avg_losses = {}
            for loss_x in self.config.LOSSES:
                avg_losses[loss_x] = running_loss_values[loss_x] / total_train_samples
                avg_losses_lists[loss_x].append(avg_losses[loss_x])
            for ind_loss_x in self.config.IND_LOSSES:
                if running_ind_loss_counts[ind_loss_x] > 0:
                    running_ind_loss_values[ind_loss_x] = running_ind_loss_values[ind_loss_x] / running_ind_loss_counts[ind_loss_x]
                else:
                    running_ind_loss_values[ind_loss_x] = float('nan')
                avg_losses_lists[ind_loss_x].append(running_ind_loss_values[ind_loss_x])
            print(f'Epoch [{epoch + 1}/{self.config.EPOCHS}], Train Mean Loss [cm]: {avg_loss:.2f}, ' + ' '.join((f'Train Mean {loss_x}: {avg_losses[loss_x]:.2f},' for loss_x in self.config.LOSSES)) + ', ' + ' '.join((f'Train Mean {ind_loss_x}: {running_ind_loss_values[ind_loss_x]:.2f},' for ind_loss_x in self.config.IND_LOSSES)))
            self.logger.info(f'Epoch [{epoch + 1}/{self.config.EPOCHS}], Train Mean Loss [cm]: {avg_loss:.2f}, ' + ' '.join((f'Train Mean {loss_x}: {avg_losses[loss_x]:.2f},' for loss_x in self.config.LOSSES)) + ', ' + ' '.join((f'Train Mean {ind_loss_x}: {running_ind_loss_values[ind_loss_x]:.2f},' for ind_loss_x in self.config.IND_LOSSES)))
            self.writer.add_scalar('Loss/Train', avg_loss, epoch)
            for loss_x in self.config.LOSSES:
                self.writer.add_scalar(f'{loss_x}/Train', avg_losses[loss_x], epoch)
            for ind_loss_x in self.config.IND_LOSSES:
                self.writer.add_scalar(f'{ind_loss_x}/Train', running_ind_loss_values[ind_loss_x], epoch)
            self.writer.add_scalar('Elapsed Time (seconds)', time.time() - start_time, epoch)
            self.model.eval()
            running_vloss = 0.0
            with torch.no_grad():
                validation_batches = iter(self.val_loader)
                if self.running_local:
                    validation_batches = tqdm(validation_batches, total=len(self.val_loader), desc='Validation')
                for (vstep, (radar, target, *rest)) in enumerate(validation_batches):
                    if rest:
                        pointcloud = rest[0]
                        vpointcloud = pointcloud.to(self.device, non_blocking=True)
                    vradar = radar.to(self.device, non_blocking=True)
                    validation_target = target.to(self.device, non_blocking=True)
                    with self._autocast_context():
                        if self._supports_trackformer_api() and validation_target.dim() == 3:
                            trackformer_input = self._select_trackformer_input(vradar, rest)
                            (voutput, vsupervised_target, trackformer_aux) = self._forward_trackformer_sequence(trackformer_input, validation_target, use_gt_matching=False)
                        else:
                            (vmodel_output, trackformer_aux) = self._forward_batch(vradar, rest)
                            voutput = self._get_supervised_output(vmodel_output, trackformer_aux)
                            vsupervised_target = self._get_supervised_target(validation_target, trackformer_aux)
                        if self.config.ADAPTIONS_FLAG:
                            for adaptions in self.config.ADAPTIONS:
                                voutput = utils.train_utils.get_adaptions(adaptions)(voutput, vsupervised_target, config=self.config)
                            if trackformer_aux is not None:
                                trackformer_aux['adapted_final_output'] = voutput
                        val_loss = self.criterion(voutput, vsupervised_target, config=self.config, validation=True, epoch=epoch, model=self.model, optimizer=self.optimizer, trackformer_aux=trackformer_aux)
                    running_vloss += val_loss.item() * vsupervised_target.size(0)
                    total_val_samples += vsupervised_target.size(0)
                    metric_voutput = voutput.float()
                    for loss_x in self.config.LOSSES:
                        vloss_value = utils.train_utils.get_losses(loss_x)(metric_voutput, vsupervised_target, config=self.config, trackformer_aux=trackformer_aux, reuse_cached_auxiliary_losses=True)
                        running_vloss_values[loss_x] = running_vloss_values[loss_x] + vloss_value * vsupervised_target.size(0)
                    for ind_loss_x in self.config.IND_TEST_LOSSES:
                        loss_value = utils.train_utils.get_losses(ind_loss_x)(metric_voutput, vsupervised_target, config=self.config, item=True, undefined_value=float('nan'), trackformer_aux=trackformer_aux)
                        loss_value = loss_value.item() if hasattr(loss_value, 'item') else loss_value
                        loss_value = float(loss_value)
                        if np.isfinite(loss_value):
                            running_ind_vloss_values[ind_loss_x] += loss_value * vsupervised_target.size(0)
                            running_ind_vloss_counts[ind_loss_x] += vsupervised_target.size(0)
            avg_vloss = running_vloss / total_val_samples
            avg_vloss_list.append(avg_vloss)
            avg_vlosses = {}
            for loss_x in self.config.LOSSES:
                avg_vlosses[loss_x] = running_vloss_values[loss_x] / total_val_samples
                avg_vlosses_lists[loss_x].append(avg_vlosses[loss_x])
            for ind_loss_x in self.config.IND_LOSSES:
                if running_ind_vloss_counts[ind_loss_x] > 0:
                    running_ind_vloss_values[ind_loss_x] = running_ind_vloss_values[ind_loss_x] / running_ind_vloss_counts[ind_loss_x]
                else:
                    running_ind_vloss_values[ind_loss_x] = float('nan')
                avg_vlosses_lists[ind_loss_x].append(running_ind_vloss_values[ind_loss_x])
            print(f'Epoch [{epoch + 1}/{self.config.EPOCHS}], Val Mean Loss [cm]: {avg_vloss:.2f}, ' + ' '.join((f'Val Mean {loss_x}: {avg_vlosses[loss_x]:.2f},' for loss_x in self.config.LOSSES)) + ' '.join((f'Val Mean {ind_loss_x}: {running_ind_vloss_values[ind_loss_x]:.2f},' for ind_loss_x in self.config.IND_LOSSES)))
            self.logger.info(f'Epoch [{epoch + 1}/{self.config.EPOCHS}], Val Mean Loss [cm]: {avg_vloss:.2f}, ' + ' '.join((f'Val Mean {loss_x}: {avg_vlosses[loss_x]:.2f},' for loss_x in self.config.LOSSES)) + ' '.join((f'Val Mean {ind_loss_x}: {running_ind_vloss_values[ind_loss_x]:.2f},' for ind_loss_x in self.config.IND_LOSSES)))
            self.writer.add_scalar('Loss/Validation', avg_vloss, epoch)
            for loss_x in self.config.LOSSES:
                self.writer.add_scalar(f'{loss_x}/Validation', avg_vlosses[loss_x], epoch)
            for ind_loss_x in self.config.IND_LOSSES:
                self.writer.add_scalar(f'{ind_loss_x}/Validation', running_ind_vloss_values[ind_loss_x], epoch)
            motr_streaming_metrics = None
            if self._use_motr_streaming_validation():
                interval = max(1, int(getattr(self.config, 'MOTR_STREAMING_VALIDATION_INTERVAL', 1)))
                if (epoch + 1) % interval == 0:
                    motr_streaming_metrics = self._run_motr_streaming_validation()
                    self._log_motr_streaming_validation(motr_streaming_metrics, epoch)
            for key in motr_streaming_val_history:
                value = None if motr_streaming_metrics is None else motr_streaming_metrics.get(key)
                motr_streaming_val_history[key].append(value)
            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_states.pth'))
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
            if self._use_motr_streaming_validation_for_best():
                if motr_streaming_metrics is not None and motr_streaming_metrics['motr_val_score'] > best_motr_val_score:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
                    best_motr_val_score = motr_streaming_metrics['motr_val_score']
                    lowest_val_loss = avg_vloss
                    epoch_best_val_loss = epoch
                    epoch_best_motr_val_score = epoch
            elif avg_vloss < lowest_val_loss:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
                lowest_val_loss = avg_vloss
                epoch_best_val_loss = epoch
            if exiting_flag:
                break
        loss_dict = {'avg_train_loss': avg_loss_list, 'avg_val_loss': avg_vloss_list, 'epoch_best_val_loss': epoch_best_val_loss, 'best_val_loss': best_val_loss, 'lowest_val_loss': lowest_val_loss, 'best_model_selection': 'motr_streaming_' + str(getattr(self.config, 'MOTR_STREAMING_VALIDATION_SELECTION_METRIC', 'score')).lower() if self._use_motr_streaming_validation_for_best() else 'val_loss', 'best_motr_val_score': None if best_motr_val_score == float('-inf') else best_motr_val_score, 'epoch_best_motr_val_score': epoch_best_motr_val_score}
        for (key, values) in motr_streaming_val_history.items():
            loss_dict[f'motr_streaming_val_{key}'] = values
        for loss_x in self.config.LOSSES:
            loss_dict[f'train_{loss_x}'] = avg_losses_lists[loss_x]
            loss_dict[f'val_{loss_x}'] = avg_vlosses_lists[loss_x]
        for ind_loss_x in self.config.IND_LOSSES:
            loss_dict[f'train_{ind_loss_x}'] = running_ind_loss_values[ind_loss_x]
            loss_dict[f'val_{ind_loss_x}'] = running_ind_loss_values[ind_loss_x]
        utils.save_metrics.save_metrics_dict_to_json(loss_dict, os.path.join(self.save_path, 'train_val_metrics.json'))
        utils.train_utils.plot_and_save_train_val_loss(avg_loss_list, avg_vloss_list, filename=self.save_path)
        return avg_vloss
