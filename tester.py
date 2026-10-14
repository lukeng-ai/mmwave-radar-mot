import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import utils.train_utils
import utils.save_metrics
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import os

class Tester:

    def __init__(self, model, test_loader, criterion, config, logger=(None, None), save_path='./', gif_steps=1):
        self.config = config
        self.device = self.config.DEVICE
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.logger = logger
        self.save_path = save_path
        self.gif_steps = gif_steps
        os.makedirs(self.save_path, exist_ok=True)
        self.predict_tensor = torch.tensor([])
        self.ground_truth_tensor = torch.tensor([])
        if logger[0] is None:
            logging.basicConfig(filename=os.path.join(self.save_path, 'logger.log'), encoding='utf-8', level=logging.INFO, format='{asctime} - {levelname} - {message}', style='{', datefmt='%Y-%m-%d %H:%M')
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger[0]
        if logger[1] is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_path, 'tensorboard_logs'))
        else:
            self.writer = logger[1]

    def _get_supervised_target(self, target, output=None):
        if target.dim() == 3:
            (batch_size, seq_length) = target.shape[:2]
            if output is not None and output.dim() >= 2 and (output.shape[0] == batch_size * seq_length):
                return target.reshape(batch_size * seq_length, -1)
            return target[:, -1]
        return target

    def _get_last_frame_output(self, output, target):
        if target.dim() != 3 or output.dim() < 2:
            return output
        (batch_size, seq_length) = target.shape[:2]
        if output.shape[0] != batch_size * seq_length:
            return output
        return output.reshape(batch_size, seq_length, -1)[:, -1]

    def _get_trackformer_model(self):
        if callable(getattr(self.model, 'forward_frame', None)):
            return self.model
        wrapped_model = getattr(self.model, 'model', None)
        if callable(getattr(wrapped_model, 'forward_frame', None)):
            return wrapped_model
        return None

    def _is_motr_model(self):
        trackformer_model = self._get_trackformer_model()
        if trackformer_model is None:
            return False
        return str(getattr(trackformer_model, 'tracking_approach', '')).lower() == 'motr'

    def _is_streaming_tracking_model(self):
        trackformer_model = self._get_trackformer_model()
        if trackformer_model is None:
            return False
        return str(getattr(trackformer_model, 'tracking_approach', '')).lower() in ('motr', 'trackformer', 'bytetrack')

    def _flatten_frame_predictions(self, regression, classification):
        regression = regression.reshape(regression.shape[0], -1)
        classification = classification.reshape(classification.shape[0], -1)
        return torch.cat((regression, classification), dim=-1)

    def _tracking_result_path(self):
        folder_name = str(getattr(self.config, 'TRACKING_TEST_OUTPUT_DIR', 'test_tracking'))
        return os.path.join(self.save_path, folder_name)

    def _recording_sequence_lengths(self, total_frames):
        lengths = getattr(self.test_loader.dataset, 'recording_sequence_lengths', None)
        if lengths is None:
            return [int(total_frames)]
        lengths = [int(length) for length in lengths if int(length) > 0]
        if sum(lengths) != int(total_frames):
            self.logger.warning('Test recording lengths do not match predictions (%s != %s); using one continuous sequence.', sum(lengths), total_frames)
            return [int(total_frames)]
        return lengths

    def _localization_tracking_arrays(self):
        num_people = int(getattr(self.config, 'PNT_NUM_PEOPLE', 2))
        regression_parameter = int(getattr(self.config, 'REGRESSION_PARAMETER', 6))
        num_regression_values = regression_parameter * num_people
        if self.predict_tensor.shape[1] < num_regression_values + num_people:
            raise ValueError('Tracking test expects localization output containing regression and classification values.')
        pred_regression = torch.zeros((self.predict_tensor.shape[0], num_people, 6), dtype=torch.float32)
        copy_dims = min(regression_parameter, 6)
        raw_regression = self.predict_tensor[:, :num_regression_values].reshape(-1, num_people, regression_parameter)
        pred_regression[:, :, :copy_dims] = raw_regression[:, :, :copy_dims]
        pred_scores = torch.sigmoid(self.predict_tensor[:, num_regression_values:num_regression_values + num_people])
        (gt_regression_raw, gt_classification) = utils.train_utils.extract_localization_targets(self.ground_truth_tensor, self.config)
        gt_regression = torch.zeros((self.ground_truth_tensor.shape[0], num_people, 6), dtype=torch.float32)
        gt_copy_dims = min(gt_regression_raw.shape[-1], 6)
        gt_regression[:, :, :gt_copy_dims] = gt_regression_raw[:, :, :gt_copy_dims]
        gt_valid = gt_classification >= 0.5
        return (pred_regression.numpy(), pred_scores.numpy(), gt_regression.numpy(), gt_valid.numpy())

    def _run_kalman_tracking_test(self):
        from hlat.utils.kalman import evaluate_kalman_tracking, save_kalman_tracking_result
        (pred_regression, pred_scores, gt_regression, gt_valid) = self._localization_tracking_arrays()
        (metrics, outputs) = evaluate_kalman_tracking(pred_regression, pred_scores, gt_regression, gt_valid, sequence_lengths=self._recording_sequence_lengths(pred_regression.shape[0]), var_process_noise=float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_VAR_PROCESS_NOISE', 0.14)), var_measurement_noise=float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_VAR_MEASUREMENT_NOISE', 3.0)), assignment_threshold=float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_ASSIGNMENT_THRESH', 0.4)), max_miss=int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MAX_MISS', 6)), max_active_filters=int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MAX_ACTIVE_FILTERS', getattr(self.config, 'PNT_NUM_PEOPLE', 2))), filter_short_frames=int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_FILTER_SHORT_FRAMES', 0)), kalman_dim=int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_DIM', getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MATCH_DIMS', 2))), kalman_pred_score_threshold=float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_PRED_SCORE_THRESH', 0.5)), tracking_score_threshold=float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_SCORE_THRESH', 0.5)), tracking_match_threshold=float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MATCH_THRESH', 1.0)), tracking_match_dims=int(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_MATCH_DIMS', 2)), stabilize_gt=bool(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_STABILIZE_GT', False)), gt_stabilization_threshold=float(getattr(self.config, 'KALMAN_TRACKING_VALIDATION_GT_STABILIZATION_THRESH', 0.2)), evaluation_y_limit=getattr(self.config, 'TRACKING_TEST_Y_LIMIT', 4.9), reset_between_sequences=bool(getattr(self.config, 'TRACKING_TEST_RESET_BETWEEN_SEQUENCES', True)))
        save_kalman_tracking_result(self._tracking_result_path(), metrics, outputs)
        return metrics

    def _run_streaming_tracking_test(self):
        from hlat.utils.evaluate_tba_inference import evaluate_motr_streaming, get_default_score_threshold
        from hlat.utils.tracking_metrics import save_tracking_metrics
        tracking_model = self._get_trackformer_model()
        score_threshold = getattr(self.config, 'MOTR_STREAMING_VALIDATION_SCORE_THRESH', None)
        if score_threshold is None:
            score_threshold = get_default_score_threshold(self.config)
        (metrics, saved, _, _) = evaluate_motr_streaming(tracking_model, self.test_loader.dataset, self.config, score_threshold=float(score_threshold), match_threshold=float(getattr(self.config, 'MOTR_STREAMING_VALIDATION_MATCH_THRESH', 1.0)), match_dims=int(getattr(self.config, 'MOTR_STREAMING_VALIDATION_MATCH_DIMS', 2)), stabilize_gt=bool(getattr(self.config, 'MOTR_STREAMING_VALIDATION_STABILIZE_GT', False)), gt_stabilization_threshold=getattr(self.config, 'MOTR_STREAMING_VALIDATION_GT_STABILIZATION_THRESH', 0.2), save_preds=True, evaluate_warmup=bool(getattr(self.config, 'MOTR_STREAMING_VALIDATION_EVALUATE_WARMUP_FLAG', False)), context_replay=False, evaluation_y_limit=getattr(self.config, 'TRACKING_TEST_Y_LIMIT', 4.9))
        metrics['evaluation_y_limit_mode'] = 'ignore'
        metrics['frames_over_evaluation_y_limit'] = int(metrics.get('num_ignored_frames', 0))
        result_path = self._tracking_result_path()
        os.makedirs(result_path, exist_ok=True)
        output_names = {'pred_regression': 'preds_regression', 'pred_scores': 'preds_classification', 'gt_regression': 'targets_regression', 'gt_valid': 'targets_classification', 'pred_track_ids': 'pred_track_ids', 'evaluation_valid_mask': 'evaluation_valid_mask'}
        for (source_name, output_name) in output_names.items():
            if saved is not None and source_name in saved:
                np.save(os.path.join(result_path, f'{output_name}.npy'), saved[source_name])
        np.save(os.path.join(result_path, 'recording_sequence_lengths.npy'), np.asarray(metrics.get('recording_sequence_counts', []), dtype=np.int64))
        if saved is not None:
            np.savez_compressed(os.path.join(result_path, 'streaming_predictions.npz'), **saved)
        save_tracking_metrics(metrics, os.path.join(result_path, 'tracking_metrics.json'))
        return metrics

    def _run_tracking_test(self):
        if not bool(getattr(self.config, 'TRACKING_TEST_FLAG', True)):
            return None
        if not (bool(getattr(self.config, 'LOCALIZATION_FLAG', False)) or self._is_streaming_tracking_model()):
            return None
        if self._is_streaming_tracking_model():
            metrics = self._run_streaming_tracking_test()
        else:
            metrics = self._run_kalman_tracking_test()
        msg = f"Streaming tracking test: HOTA={metrics.get('hota', 0.0):.4f}, MOTA={metrics.get('mota') or 0.0:.4f}, DetA={metrics.get('deta', metrics.get('detection_accuracy', 0.0)):.4f}, AssA={metrics.get('assa', 0.0):.4f}, IDSW={metrics.get('id_switches', 0)}"
        print(msg)
        self.logger.info(msg)
        for key in ('hota', 'mota', 'deta', 'assa', 'precision', 'recall', 'f1', 'mae_xy', 'rmse_xy', 'mae_xyz', 'rmse_xyz', 'id_switches', 'track_fragmentations'):
            value = metrics.get(key)
            if value is not None:
                self.writer.add_scalar(f'TrackingTest/{key}', float(value), 1)
        return metrics

    def test(self):
        self.model.eval()
        running_mean_loss = 0.0
        running_loss = 0.0
        losses_dict = {}
        ind_losses_dict = {}
        total_samples = 0
        self.predict_tensor = torch.tensor([])
        self.ground_truth_tensor = torch.tensor([])
        with torch.no_grad():
            for (step, (radar, target, *rest)) in enumerate(self.test_loader):
                if rest:
                    pointcloud = rest[0]
                    pointcloud = pointcloud.to(self.device, non_blocking=True)
                    rest = (pointcloud,)
                radar = radar.to(self.device)
                target = target.to(self.device)
                trackformer_model = self._get_trackformer_model()
                if trackformer_model is not None:
                    trackformer_input = rest[0] if rest else radar
                    (_, aux) = trackformer_model(trackformer_input, return_aux=True)
                    output = self._flatten_frame_predictions(aux['pred_regression_per_frame'][:, -1], aux['pred_classification_per_frame'][:, -1])
                elif rest:
                    output = self.model((radar, pointcloud))
                else:
                    output = self.model(radar)
                output = self._get_last_frame_output(output, target)
                supervised_target = self._get_supervised_target(target)
                if self.config.ADAPTIONS_FLAG:
                    for adaptions in self.config.ADAPTIONS:
                        output = utils.train_utils.get_adaptions(adaptions)(output, supervised_target, config=self.config)
                self.predict_tensor = torch.concat((self.predict_tensor, output.cpu().detach()))
                self.ground_truth_tensor = torch.concat((self.ground_truth_tensor, supervised_target.cpu().detach()))
                loss = self.criterion(output, supervised_target, config=self.config, test=True, model=self.model)
                running_loss += loss.item() * supervised_target.size(0)
                total_samples += supervised_target.size(0)
            for loss_x in self.config.TEST_LOSSES:
                loss_value = utils.train_utils.get_losses(loss_x)(self.predict_tensor, self.ground_truth_tensor, config=self.config)
                if isinstance(loss_value, list):
                    if not losses_dict.get(loss_x):
                        losses_dict[loss_x] = []
                    losses_dict[loss_x].append(loss_value)
                else:
                    if not losses_dict.get(loss_x):
                        losses_dict[loss_x] = 0
                    losses_dict[loss_x] = losses_dict[loss_x] + loss_value
            for ind_loss_x in self.config.IND_TEST_LOSSES:
                loss_value = utils.train_utils.get_losses(ind_loss_x)(self.predict_tensor, self.ground_truth_tensor, config=self.config, item=True, ind_loss_x=ind_losses_dict)
                if not isinstance(loss_value, list):
                    ind_losses_dict[ind_loss_x] = loss_value
            avg_loss = running_loss / total_samples
            print(f'Test Mean Loss: {avg_loss:.2f}, ' + ', '.join((f'Test Mean {loss_x}: {losses_dict[loss_x]:.2f}' for loss_x in self.config.TEST_LOSSES if not isinstance(losses_dict[loss_x], list))) + ', ' + ', '.join((f'Test Mean {ind_loss_x}: {ind_losses_dict[ind_loss_x]:.2f}' for ind_loss_x in self.config.IND_TEST_LOSSES if not isinstance(ind_losses_dict[ind_loss_x], list))))
            self.logger.info(f'Test Mean Loss: {avg_loss:.2f}, ' + ' '.join((f'Test Mean {loss_x}: {losses_dict[loss_x]:.2f},' for loss_x in self.config.TEST_LOSSES if not isinstance(losses_dict[loss_x], list))) + ', ' + ', '.join((f'Test Mean {ind_loss_x}: {ind_losses_dict[ind_loss_x]:.2f}' for ind_loss_x in self.config.IND_TEST_LOSSES if not isinstance(ind_losses_dict[ind_loss_x], list))))
            self.writer.add_scalar('Loss/Test', avg_loss, 1)
            for loss_x in self.config.TEST_LOSSES:
                if not isinstance(losses_dict[loss_x], list):
                    self.writer.add_scalar(f'{loss_x}/Test', losses_dict[loss_x], 1)
            for ind_loss_x in self.config.IND_TEST_LOSSES:
                if not isinstance(ind_losses_dict[ind_loss_x], list):
                    self.writer.add_scalar(f'{ind_loss_x}/Test', ind_losses_dict[ind_loss_x], 1)
            test_metrics_dict = {'avg_test_loss': avg_loss, 'steps': step + 1}
            for loss_x in self.config.TEST_LOSSES:
                test_metrics_dict[f'test_{loss_x}'] = losses_dict[loss_x]
            for ind_loss_x in self.config.IND_TEST_LOSSES:
                test_metrics_dict[f'test_{ind_loss_x}'] = ind_losses_dict[ind_loss_x]
            utils.save_metrics.save_metrics_dict_to_json(test_metrics_dict, os.path.join(self.save_path, 'test_metrics.json'))
            for loss_x in self.config.TEST_LOSSES:
                if loss_x.lower().startswith('step'):
                    step_test_error = torch.tensor(losses_dict[loss_x]).cpu().detach().numpy()
                    utils.train_utils.plot_and_save_test_loss(step_test_error, filename=self.save_path, label=loss_x)
            self._run_tracking_test()
        return avg_loss
