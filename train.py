import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import gc
import os
from datetime import datetime
from tqdm import tqdm
import timeit
import socket
import subprocess
import tester
import trainer
import time
import utils.train_utils
import utils.config as config
import sys
import random
from dataset import Dataset
from model import *

def _sanitize_path_component(value):
    return ''.join((char if char.isalnum() or char in ('-', '_', '.') else '-' for char in str(value)))

def build_train_name():
    now = datetime.now()
    train_name_parts = [now.strftime('%Y-%m-%d_%H-%M-%S'), now.strftime('%f')]
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        train_name_parts.append(f'job{_sanitize_path_component(slurm_job_id)}')
    else:
        train_name_parts.extend([_sanitize_path_component(socket.gethostname()), f'pid{os.getpid()}'])
    return '_'.join(train_name_parts)

def create_run_directory(result_dir, train_name_dir):
    os.makedirs(os.path.join(result_dir, train_name_dir), exist_ok=True)
    train_name = build_train_name()
    model_save_path = os.path.join(result_dir, train_name_dir, train_name)
    try:
        os.makedirs(model_save_path)
        return (train_name, model_save_path)
    except FileExistsError:
        pass
    for index in range(1, 1000):
        candidate_name = f'{train_name}_{index}'
        candidate_path = os.path.join(result_dir, train_name_dir, candidate_name)
        try:
            os.makedirs(candidate_path)
            return (candidate_name, candidate_path)
        except FileExistsError:
            continue
    raise RuntimeError(f'Could not create a unique run directory below {os.path.join(result_dir, train_name_dir)}')

def build_data_loader(dataset, cfg, shuffle):
    num_workers = int(cfg.NUM_WORKERS)
    kwargs = {'batch_size': cfg.BATCH_SIZE, 'num_workers': num_workers, 'pin_memory': cfg.DEVICE.type == 'cuda', 'shuffle': shuffle}
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(getattr(cfg, 'DATALOADER_PERSISTENT_WORKERS_FLAG', True))
        kwargs['prefetch_factor'] = int(getattr(cfg, 'DATALOADER_PREFETCH_FACTOR', 2))
    return DataLoader(dataset, **kwargs)

def run_post_test_result_generation(cfg, model_save_path, logger):
    if not bool(getattr(cfg, 'GENERATE_RESULTS_AFTER_TEST_FLAG', False)):
        return
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hlat', 'utils', 'generate_results.py')
    command = [sys.executable, script_path, '--input', model_save_path, '--datasetdir', cfg.DATA_PATH, '--plot-format', str(getattr(cfg, 'GENERATE_RESULTS_PLOT_FORMAT', 'pdf')), '--evaluation-y-limit', str(getattr(cfg, 'TRACKING_TEST_Y_LIMIT', 4.9))]
    trackformer_model = getattr(cfg, 'MODEL', '')
    is_streaming_model = any((tag in str(trackformer_model).upper() for tag in ('PNTKPCONVTBA',)))
    if is_streaming_model:
        command.extend(['--motr-score-threshold', str(getattr(cfg, 'MOTR_STREAMING_VALIDATION_SCORE_THRESH', 0.5)), '--motr-match-threshold', str(getattr(cfg, 'MOTR_STREAMING_VALIDATION_MATCH_THRESH', 1.0)), '--motr-match-dims', str(getattr(cfg, 'MOTR_STREAMING_VALIDATION_MATCH_DIMS', 2))])
    else:
        command.extend(['--kalman-var-process-noise', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_VAR_PROCESS_NOISE', 0.14)), '--kalman-var-measurement-noise', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_VAR_MEASUREMENT_NOISE', 3.0)), '--kalman-assignment-threshold', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_ASSIGNMENT_THRESH', 0.4)), '--kalman-max-miss', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_MAX_MISS', 6)), '--kalman-max-active-filters', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_MAX_ACTIVE_FILTERS', getattr(cfg, 'PNT_NUM_PEOPLE', 2))), '--kalman-filter-short-frames', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_FILTER_SHORT_FRAMES', 0)), '--kalman-dim', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_DIM', getattr(cfg, 'KALMAN_TRACKING_VALIDATION_MATCH_DIMS', 2))), '--kalman-pred-score-threshold', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_PRED_SCORE_THRESH', 0.5)), '--tracking-score-threshold', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_SCORE_THRESH', 0.5)), '--tracking-match-threshold', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_MATCH_THRESH', 1.0)), '--tracking-match-dims', str(getattr(cfg, 'KALMAN_TRACKING_VALIDATION_MATCH_DIMS', 2))])
    print('Generating selected post-test PDF results ...')
    logger.info('Post-test result command: %s', ' '.join(command))
    subprocess.run(command, check=True)

def main():
    start = timeit.default_timer()
    start_time = datetime.today()
    print(f'Starting time: {start_time}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', end='\n\n')
    HOST_NAME = socket.gethostname()
    running_local = False if HOST_NAME.__contains__('tg') or HOST_NAME.__contains__('a') else True
    print(f'running local: {running_local}')
    args = utils.train_utils.get_parser()
    cfg = config.Config(LOAD_FROM_CONFIG_FILE_FLAG=True, CONFIG_PATH=args.get('config') or './configs/tba_seq10_memory10.yaml', DATA_PATH=args.get('datasetdir') or './data', DATA_SLICE_PATH=args.get('slicesdir') or './cache/slices', NUM_WORKERS=8, DEVICE=device, MODEL='PntKPConvTbA', SEED=42, args=args)
    cfg.print_config()
    TRAIN_NAME_DIR = f'{cfg.MODEL}' if cfg.CHAR == '-' else f'{cfg.MODEL}_{cfg.CHAR}'
    work_dir = cfg.WORKDIR or os.getcwd()
    print(f'workdir: {work_dir}')
    result_dir = os.path.join(work_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    (TRAIN_NAME, model_save_path) = create_run_directory(result_dir, TRAIN_NAME_DIR)
    log_path = os.path.join(model_save_path, 'logger.log')
    print(f'run name: {TRAIN_NAME}')
    cfg.SAVE_PATH = model_save_path
    cfg.save_config(model_save_path)
    utils.train_utils.set_random_seed(cfg.SEED)
    model = globals()[cfg.MODEL](config=cfg).to(device)
    pretrained_fe_path = cfg.args.get('PRETRAINED_FE')
    if pretrained_fe_path:
        print(f'Loading pretrained feature extractor weights from {pretrained_fe_path}')
        fe_state_dict = torch.load(pretrained_fe_path, map_location=device, weights_only=True)
        model.feature_extractor.load_state_dict(fe_state_dict, strict=False)
    criterion = utils.train_utils.get_criterion(cfg.CRITERION)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    (total_params, trainable_params) = utils.train_utils.get_model_parameters(model)
    print(model)
    data = Dataset(config=cfg)
    (train_dataset, val_dataset, test_dataset, config_dataset_dict, compose_transform) = data.load_dataset()
    gif_steps = cfg.SEQUENCE_LENGTH
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.INFO, format='{asctime} - {levelname} - {message}', style='{', datefmt='%Y-%m-%d %H:%M')
    logger = logging.getLogger(__name__)
    print(config_dataset_dict, end='\n\n')
    print(compose_transform)
    logger.info(config_dataset_dict)
    logger.info(compose_transform)
    utils.train_utils.printTimestamp(start, 'DATASETTIME', logger)
    logging_params = [*vars(cfg).items(), ('criterion', criterion), ('optimizer', optimizer), ('model', model)]
    utils.train_utils.log_params(logging_params, logger)
    print(*[f'{key}: {value},' for (key, value) in vars(cfg).items()])
    print(f'Total Model Parameters: {total_params}, Trainabel Model Parameters: {trainable_params}', end='\n\n')
    logger.info(f'Total Model Parameters: {total_params}, Trainabel Model Parameters: {trainable_params}')
    tensorboad_logger = SummaryWriter(log_dir=os.path.join(model_save_path, 'tensorboard_logs'))
    train_loader = build_data_loader(train_dataset, cfg, shuffle=True)
    val_loader = build_data_loader(val_dataset, cfg, shuffle=False)
    trainer_cls = trainer.AttentionTrainer if callable(getattr(model, 'forward_frame', None)) or callable(getattr(getattr(model, 'model', None), 'forward_frame', None)) else trainer.Trainer
    training = trainer_cls(model, train_loader, val_loader, criterion, optimizer, cfg, (logger, tensorboad_logger), save_path=model_save_path, running_local=running_local)
    start_train_time = timeit.default_timer()
    training.train()
    utils.train_utils.printTimestamp(start_train_time, 'TRAINTIME', logger)
    training = None
    del train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    test_loader = build_data_loader(test_dataset, cfg, shuffle=False)
    print('Loading best model ...')
    model.load_state_dict(torch.load(os.path.join(cfg.SAVE_PATH, 'best_model.pth'), map_location=cfg.DEVICE, weights_only=True))
    testing = tester.Tester(model, test_loader, criterion, cfg, (logger, tensorboad_logger), model_save_path, gif_steps)
    start_test_time = timeit.default_timer()
    testing.test()
    utils.train_utils.printTimestamp(start_test_time, 'TESTTIME', logger)
    run_post_test_result_generation(cfg, model_save_path, logger)
    tensorboad_logger.close()
    utils.train_utils.printTimestamp(start, 'RUNTIME', logger)
if __name__ == '__main__':
    main()
