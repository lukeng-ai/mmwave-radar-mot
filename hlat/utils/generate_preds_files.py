import numpy as np
import os
import sys
import torch
import yaml
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import time
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model import *
import utils.train_utils
import utils.config
from dataset import Dataset
from hlat.utils.tracking_metrics import compute_tracking_metrics_from_arrays, save_tracking_metrics
from hlat.utils.prediction_outputs import save_preds_other_if_enabled

def unknown_constructor(loader, node):
    print(f"Warning: Unknown YAML tag '{node.tag}' encountered. Using string representation.")
    return str(node.value)
yaml.add_constructor(None, unknown_constructor, Loader=yaml.SafeLoader)

def load_model(best_model_dir, cfg):
    model = globals()[cfg.MODEL](config=cfg).to(cfg.DEVICE)
    model.load_state_dict(torch.load(os.path.join(best_model_dir, 'best_model.pth'), map_location=cfg.DEVICE, weights_only=True))
    model.to(cfg.DEVICE)
    model.eval()
    return model

def _get_trackformer_model(model):
    if callable(getattr(model, 'forward_frame', None)):
        return model
    wrapped_model = getattr(model, 'model', None)
    if callable(getattr(wrapped_model, 'forward_frame', None)):
        return wrapped_model
    return None

def _is_motr_model(model):
    trackformer_model = _get_trackformer_model(model)
    if trackformer_model is None:
        return False
    return str(getattr(trackformer_model, 'tracking_approach', '')).lower() == 'motr'

def _select_trackformer_input(radar, rest):
    if rest:
        return rest[0]
    return radar

def _flatten_frame_predictions(regression, classification):
    regression = regression.reshape(regression.shape[0], -1)
    classification = classification.reshape(classification.shape[0], -1)
    return torch.cat((regression, classification), dim=-1)

def _get_supervised_target(target):
    if target.dim() == 3:
        return target[:, -1]
    return target

def _match_output_to_supervised_target(output, supervised_target, cfg):
    if output.shape[0] == supervised_target.shape[0]:
        return output
    sequence_length = getattr(cfg, 'SEQUENCE_LENGTH', None)
    if sequence_length and output.shape[0] == supervised_target.shape[0] * sequence_length:
        return output.reshape(supervised_target.shape[0], sequence_length, -1)[:, -1]
    raise RuntimeError(f'Model output batch dimension does not match target batch dimension: output has {output.shape[0]} rows, target has {supervised_target.shape[0]} rows.')

def _load_regression_array(path, cls_shape):
    regression = np.load(path)
    if regression.ndim == 3:
        return regression
    if regression.ndim == 2:
        return regression.reshape(regression.shape[0], cls_shape[1], -1)
    raise ValueError(f'Unsupported regression shape in {path}: {regression.shape}')

def generate_tracking_metrics_from_files(result_path, score_threshold=0.5, match_threshold=1.0, match_dims=2, stabilize_gt=False, gt_stabilization_threshold=None):
    pred_cls = np.load(os.path.join(result_path, 'preds_classification.npy'))
    gt_cls = np.load(os.path.join(result_path, 'targets_classification.npy'))
    pred_reg = _load_regression_array(os.path.join(result_path, 'preds_regression.npy'), pred_cls.shape)
    gt_reg = _load_regression_array(os.path.join(result_path, 'targets_regression.npy'), gt_cls.shape)
    tracking_metrics = compute_tracking_metrics_from_arrays(pred_regression=pred_reg, pred_scores=pred_cls, gt_regression=gt_reg, gt_valid=gt_cls >= 0.5, score_threshold=score_threshold, match_threshold=match_threshold, match_dims=match_dims, pred_track_ids=np.arange(pred_reg.shape[1], dtype=np.int64), scores_are_logits=False, stabilize_gt=stabilize_gt, gt_stabilization_threshold=gt_stabilization_threshold)
    tracking_metrics['evaluation_mode'] = 'saved_dl_slots'
    tracking_metrics['result_path'] = result_path
    save_tracking_metrics(tracking_metrics, os.path.join(result_path, 'tracking_metrics.json'))
    return tracking_metrics

def predict(model, input_data, cfg, result_path):
    model.eval()
    running_mean_loss = 0.0
    running_loss = 0.0
    predict_tensor = torch.tensor([])
    ground_truth_tensor = torch.tensor([])
    losses_dict = {}
    ind_losses_dict = {}
    total_samples = 0
    total_inference_time = 0.0
    num_batches = 0
    if cfg.DEVICE == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for (step, (radar, target, *rest)) in enumerate(input_data):
            if rest:
                pointcloud = rest[0]
                pointcloud = pointcloud.to(cfg.DEVICE, non_blocking=True)
            radar = radar.to(cfg.DEVICE)
            target = target.to(cfg.DEVICE)
            if cfg.DEVICE == 'cuda':
                torch.cuda.synchronize()
                starter.record()
            else:
                start_time = time.time()
            trackformer_model = _get_trackformer_model(model)
            supervised_target = _get_supervised_target(target)
            if trackformer_model is not None:
                trackformer_input = _select_trackformer_input(radar, rest)
                (_, aux) = trackformer_model(trackformer_input, return_aux=True)
                output = _flatten_frame_predictions(aux['pred_regression_per_frame'][:, -1], aux['pred_classification_per_frame'][:, -1])
            elif rest:
                output = model((radar, pointcloud))
            else:
                output = model(radar)
            output = _match_output_to_supervised_target(output, supervised_target, cfg)
            if cfg.DEVICE == 'cuda':
                ender.record()
                torch.cuda.synchronize()
                batch_time = starter.elapsed_time(ender) / 1000.0
            else:
                batch_time = time.time() - start_time
            total_inference_time += batch_time
            num_batches += 1
            if cfg.ADAPTIONS_FLAG:
                for adaptions in cfg.ADAPTIONS:
                    output = utils.train_utils.get_adaptions(adaptions)(output, supervised_target, config=cfg)
            predict_tensor = torch.concat((predict_tensor, output.cpu().detach()))
            ground_truth_tensor = torch.concat((ground_truth_tensor, supervised_target.cpu().detach()))
            loss = utils.train_utils.get_criterion(cfg.CRITERION)(output, supervised_target, config=cfg)
            running_loss += loss.item() * supervised_target.size(0)
            total_samples += supervised_target.size(0)
        for loss_x in cfg.TEST_LOSSES:
            loss_value = utils.train_utils.get_losses(loss_x)(predict_tensor, ground_truth_tensor, config=cfg)
            if isinstance(loss_value, list):
                if not losses_dict.get(loss_x):
                    losses_dict[loss_x] = []
                losses_dict[loss_x].append(loss_value)
            else:
                if not losses_dict.get(loss_x):
                    losses_dict[loss_x] = 0
                losses_dict[loss_x] = losses_dict[loss_x] + loss_value
        for ind_loss_x in cfg.IND_TEST_LOSSES:
            loss_value = utils.train_utils.get_losses(ind_loss_x)(predict_tensor, ground_truth_tensor, config=cfg, item=True, ind_loss_x=ind_losses_dict)
            if not isinstance(loss_value, list):
                ind_losses_dict[ind_loss_x] = loss_value
        avg_loss = running_loss / total_samples
        if num_batches > 0:
            avg_inference_time = total_inference_time / num_batches
            fps = cfg.BATCH_SIZE / avg_inference_time
            print(f'\nInference Statistics:')
            print(f'  Total batches: {num_batches}')
            print(f'  Total inference time: {total_inference_time:.4f}s')
            print(f'  Average time per batch: {avg_inference_time:.4f}s')
            print(f'  Average time per sample: {avg_inference_time / cfg.BATCH_SIZE:.4f}s')
            print(f'  Throughput (FPS): {fps:.2f}\n')
        print(f'Test Mean Loss: {avg_loss:.4f}, ' + ', '.join((f'Test Mean {loss_x}: {losses_dict[loss_x]:.4f}' for loss_x in cfg.TEST_LOSSES if not isinstance(losses_dict[loss_x], list))) + ', ' + ', '.join((f'Test Mean {ind_loss_x}: {ind_losses_dict[ind_loss_x]:.4f}' for ind_loss_x in cfg.IND_TEST_LOSSES if not isinstance(ind_losses_dict[ind_loss_x], list))))
        predict_tensor_reg = torch.zeros((predict_tensor.shape[0], 6 * cfg.PNT_NUM_PEOPLE), dtype=torch.float32).reshape(-1, cfg.PNT_NUM_PEOPLE, 6)
        predict_tensor_reg[:, :cfg.PNT_NUM_PEOPLE, 0:cfg.REGRESSION_PARAMETER] = predict_tensor[:, 0:cfg.REGRESSION_PARAMETER * cfg.PNT_NUM_PEOPLE].reshape(-1, cfg.PNT_NUM_PEOPLE, cfg.REGRESSION_PARAMETER)
        np.save(os.path.join(result_path, 'targets_regression.npy'), ground_truth_tensor[:, 0:6 * cfg.PNT_NUM_PEOPLE].reshape(-1, cfg.PNT_NUM_PEOPLE, 6).cpu().numpy())
        np.save(os.path.join(result_path, 'targets_classification.npy'), ground_truth_tensor[:, 30:30 + cfg.PNT_NUM_PEOPLE].cpu().numpy())
        np.save(os.path.join(result_path, 'preds_regression.npy'), predict_tensor_reg.cpu().numpy())
        np.save(os.path.join(result_path, 'preds_classification.npy'), nn.Sigmoid()(predict_tensor[:, cfg.REGRESSION_PARAMETER * cfg.PNT_NUM_PEOPLE:cfg.REGRESSION_PARAMETER * cfg.PNT_NUM_PEOPLE + cfg.PNT_NUM_PEOPLE]).cpu().numpy())
        save_preds_other_if_enabled(result_path, predict_tensor, cfg)
        recording_sequence_lengths = getattr(getattr(input_data, 'dataset', None), 'recording_sequence_lengths', None)
        if recording_sequence_lengths is not None:
            recording_sequence_lengths = np.asarray(recording_sequence_lengths, dtype=np.int64)
            if recording_sequence_lengths.sum() == predict_tensor.shape[0]:
                np.save(os.path.join(result_path, 'recording_sequence_lengths.npy'), recording_sequence_lengths)
        tracking_metrics = generate_tracking_metrics_from_files(result_path=result_path, score_threshold=float(getattr(cfg, 'TRACKING_SCORE_THRESHOLD', 0.5)), match_threshold=float(getattr(cfg, 'TRACKING_MATCH_THRESHOLD', 1.0)), match_dims=int(getattr(cfg, 'TRACKING_MATCH_DIMS', 2)), stabilize_gt=bool(getattr(cfg, 'TRACKING_STABILIZE_GT', False)), gt_stabilization_threshold=getattr(cfg, 'TRACKING_GT_STABILIZATION_THRESHOLD', None))
        print(f"Tracking metrics: F1={tracking_metrics['f1']:.4f}, MOTA={tracking_metrics['mota']:.4f}, HOTA={tracking_metrics['hota']:.4f}, AssA={tracking_metrics['assa']:.4f}, MPE_xy={(tracking_metrics['mpe_xy'] if tracking_metrics['mpe_xy'] is not None else float('nan')):.4f}, RMSE_xy={(tracking_metrics['rmse_xy'] if tracking_metrics['rmse_xy'] is not None else float('nan')):.4f}")
    return avg_loss

def generate_metrics(result_path, cfg):
    preds_regression = np.load(os.path.join(result_path, 'preds_regression.npy'))
    if getattr(cfg, 'ADAPTIONS_FLAG', False) and 'ApplyPositionDependentXyOffset' in getattr(cfg, 'ADAPTIONS', []):
        delta_x = preds_regression[:, :, 0] / (preds_regression[:, :, 1] + 0.001) / np.sqrt(1 + (preds_regression[:, :, 0] / (preds_regression[:, :, 1] + 0.001)) ** 2) * 0.12
        delta_y = 1 / np.sqrt(1 + (preds_regression[:, :, 0] / (preds_regression[:, :, 1] + 0.001)) ** 2) * 0.12
        preds_regression = preds_regression + np.concatenate((delta_x[:, :, None], delta_y[:, :, None], np.zeros((preds_regression.shape[0], preds_regression.shape[1], preds_regression.shape[2] - 2))), axis=-1)
    preds_regression = preds_regression.reshape(-1, 6 * cfg.PNT_NUM_PEOPLE)
    preds_classification = np.load(os.path.join(result_path, 'preds_classification.npy'))
    predictions = np.concatenate((preds_regression, preds_classification), axis=-1)
    ground_truth_regression = np.load(os.path.join(result_path, 'targets_regression.npy')).reshape(-1, 6 * cfg.PNT_NUM_PEOPLE)
    ground_truth_classification = np.load(os.path.join(result_path, 'targets_classification.npy'))
    ground_truth = np.zeros((ground_truth_regression.shape[0], 99))
    ground_truth[:, 0:6 * cfg.PNT_NUM_PEOPLE] = ground_truth_regression[:, 0:6 * cfg.PNT_NUM_PEOPLE]
    ground_truth[:, 30:30 + cfg.PNT_NUM_PEOPLE] = ground_truth_classification
    predict_tensor = torch.tensor(predictions)
    ground_truth_tensor = torch.tensor(ground_truth)
    losses_dict = {}
    ind_losses_dict = {}
    avg_loss = 0
    for loss_x in cfg.TEST_LOSSES:
        loss_value = utils.train_utils.get_losses(loss_x)(predict_tensor, ground_truth_tensor, config=cfg)
        if isinstance(loss_value, list):
            if not losses_dict.get(loss_x):
                losses_dict[loss_x] = []
            losses_dict[loss_x].append(loss_value)
        else:
            if not losses_dict.get(loss_x):
                losses_dict[loss_x] = 0
            losses_dict[loss_x] = losses_dict[loss_x] + loss_value
    for ind_loss_x in cfg.IND_TEST_LOSSES:
        loss_value = utils.train_utils.get_losses(ind_loss_x)(predict_tensor, ground_truth_tensor, config=cfg, item=True, ind_loss_x=ind_losses_dict)
        if not isinstance(loss_value, list):
            ind_losses_dict[ind_loss_x] = loss_value
    tracking_metrics = generate_tracking_metrics_from_files(result_path=result_path, score_threshold=float(getattr(cfg, 'TRACKING_SCORE_THRESHOLD', 0.5)), match_threshold=float(getattr(cfg, 'TRACKING_MATCH_THRESHOLD', 1.0)), match_dims=int(getattr(cfg, 'TRACKING_MATCH_DIMS', 2)), stabilize_gt=bool(getattr(cfg, 'TRACKING_STABILIZE_GT', False)), gt_stabilization_threshold=getattr(cfg, 'TRACKING_GT_STABILIZATION_THRESHOLD', None))
    print(f'Test Mean Loss: {avg_loss:.4f}, ' + ', '.join((f'Test Mean {loss_x}: {losses_dict[loss_x]:.4f}' for loss_x in cfg.TEST_LOSSES if not isinstance(losses_dict[loss_x], list))) + ', ' + ', '.join((f'Test Mean {ind_loss_x}: {ind_losses_dict[ind_loss_x]:.4f}' for ind_loss_x in cfg.IND_TEST_LOSSES if not isinstance(ind_losses_dict[ind_loss_x], list))))
    print(f"Tracking metrics: F1={tracking_metrics['f1']:.4f}, MOTA={tracking_metrics['mota']:.4f}, HOTA={tracking_metrics['hota']:.4f}, AssA={tracking_metrics['assa']:.4f}, MPE_xy={(tracking_metrics['mpe_xy'] if tracking_metrics['mpe_xy'] is not None else float('nan')):.4f}, RMSE_xy={(tracking_metrics['rmse_xy'] if tracking_metrics['rmse_xy'] is not None else float('nan')):.4f}")
    metrics = {'avg_loss': avg_loss, 'losses': {key: value.item() if hasattr(value, 'item') else value for (key, value) in losses_dict.items()}, 'ind_losses': {key: value.item() if hasattr(value, 'item') else value for (key, value) in ind_losses_dict.items()}, 'tracking_metrics': tracking_metrics, 'result_path': result_path}
    with open(os.path.join(result_path, 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    return metrics
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-pdp', '--plot_data_path', default=None, help='Recording path relative to the dataset root, e.g. /test/recording_028. Default uses the configured test split.')
    parser.add_argument('-sp', '--save_path_name', default='all')
    parser.add_argument('-d', '--datasetdir', default=None)
    parser.add_argument('-ml', action='store_true', help='If -ml flag, then use ml processing')
    parser.add_argument('--tracking-score-threshold', type=float, default=0.5)
    parser.add_argument('--tracking-match-threshold', type=float, default=1.0)
    parser.add_argument('--tracking-match-dims', type=int, choices=[2, 3], default=2)
    parser.add_argument('--stabilize-gt', action='store_true')
    parser.add_argument('--gt-stabilization-threshold', type=float, default=0.2)
    args = parser.parse_args()
    path = args.input
    if True:
        config_path = os.path.join(path, 'config.yaml')
        cfg = utils.config.Config(LOAD_FROM_CONFIG_FILE_FLAG=True, CONFIG_PATH=config_path, DEVICE='cuda')
        cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.BATCH_SIZE = 128
        model = load_model(path, cfg)
        if args.datasetdir is not None:
            cfg.DATA_PATH = args.datasetdir
        if args.plot_data_path is not None:
            cfg.DATASET['test'] = [args.plot_data_path]
        cfg.LOCALIZATION_FLAG = True
        cfg.LOC_LOSS_SORT_BY_POSITION_FLAG = False
        cfg.TRACKING_SCORE_THRESHOLD = args.tracking_score_threshold
        cfg.TRACKING_MATCH_THRESHOLD = args.tracking_match_threshold
        cfg.TRACKING_MATCH_DIMS = args.tracking_match_dims
        cfg.TRACKING_STABILIZE_GT = args.stabilize_gt
        cfg.TRACKING_GT_STABILIZATION_THRESHOLD = args.gt_stabilization_threshold
        data = Dataset(config=cfg)
        (train_dataset, val_dataset, test_dataset, config_dataset_dict, compose_transform) = data.load_dataset(only_test_set=True)
        test_loader = DataLoader(test_dataset, cfg.BATCH_SIZE, num_workers=int(os.environ.get('GENERATE_PREDS_NUM_WORKERS', '1')), pin_memory=True, shuffle=False)
        result_path = os.path.join(path, args.save_path_name)
        os.makedirs(os.path.join(result_path), exist_ok=True)
        avg_loss = predict(model, test_loader, cfg, result_path)
