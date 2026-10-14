import subprocess
import os
import sys
import argparse
import yaml
MOTR_EVAL_ARGS = []
TRACKING_EVAL_ARGS = []
OUTPUT_SUFFIX = ''
DATASET_DIR = None
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def is_motr_result(path):
    config_path = os.path.join(path, 'config.yaml')
    if not os.path.isfile(config_path):
        return False
    with open(config_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped = line.strip()
            if stripped.upper().startswith('MODEL:'):
                model_name = stripped.upper()
                return any((tag in model_name for tag in ('PNTKPCONVTBA',)))
    return False

def build_prediction_command(path, plot_data_path, save_path_name, use_motr):
    script_name = 'evaluate_tba_inference.py' if use_motr else 'generate_preds_files.py'
    cmd = [sys.executable, os.path.join(REPO_ROOT, 'hlat', 'utils', script_name), '-i', os.path.expanduser(os.path.join(path)), '-pdp', plot_data_path, '-sp', result_name(save_path_name)]
    if DATASET_DIR is not None:
        cmd.extend(['-d', DATASET_DIR])
    if not use_motr:
        cmd.insert(4, '-ml')
        cmd.extend(TRACKING_EVAL_ARGS)
    else:
        cmd.extend(MOTR_EVAL_ARGS)
    return cmd

def visualization_script(use_motr):
    script_name = 'plot_tba_tracks.py' if use_motr else 'kalman.py'
    return os.path.join(REPO_ROOT, 'hlat', 'utils', script_name)

def result_name(name):
    return f'{name}{OUTPUT_SUFFIX}' if OUTPUT_SUFFIX else name

def prediction_result_path(cmd):
    if not any((part.endswith('generate_preds_files.py') for part in cmd)):
        return None
    if '-i' not in cmd or '-sp' not in cmd:
        return None
    input_path = cmd[cmd.index('-i') + 1]
    save_path_name = cmd[cmd.index('-sp') + 1]
    return os.path.join(input_path, save_path_name)

def prediction_outputs_exist(result_path):
    required_files = ('preds_regression.npy', 'preds_classification.npy', 'targets_regression.npy', 'targets_classification.npy', 'tracking_metrics.json')
    return all((os.path.isfile(os.path.join(result_path, name)) for name in required_files))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-d', '--datasetdir', default=None)
    parser.add_argument('--output-suffix', default='', help='Append a suffix to every generated result folder, e.g. _win or _stream.')
    parser.add_argument('--plot-format', choices=('png', 'pdf'), default='png', help='Output format for the selected trajectory plots.')
    parser.add_argument('--force-regenerate-preds', action='store_true', help='Regenerate prediction npy files even when a complete result folder already exists.')
    parser.add_argument('--motr-window-batch-eval', action='store_true', help='For MOTR results, evaluate each sequence window independently instead of stateful streaming.')
    parser.add_argument('--motr-context-replay', action='store_true', help='For MOTR results, replay each overlapping window with carried track state in streaming mode.')
    parser.add_argument('--motr-score-threshold', type=float, default=None, help='For MOTR results, override the score threshold used by evaluate_tba_inference.py.')
    parser.add_argument('--motr-keep-track-score-threshold', type=float, default=None, help='For MOTR results, override MOTR_KEEP_TRACK_SCORE_THRESH during inference.')
    parser.add_argument('--motr-new-track-score-threshold', type=float, default=None, help='For MOTR results, override MOTR_NEW_TRACK_SCORE_THRESH during inference.')
    parser.add_argument('--motr-max-inactive-steps', type=int, default=None, help='For MOTR results, override MOTR_MAX_INACTIVE_STEPS during inference.')
    parser.add_argument('--motr-promotion-suppression-radius', type=float, default=None, help='For MOTR results, override duplicate-suppression radius for promoted object queries.')
    parser.add_argument('--motr-match-threshold', type=float, default=None, help='For MOTR results, override the distance threshold used for metric matching and slot alignment.')
    parser.add_argument('--motr-match-dims', type=int, default=None, choices=(2, 3), help='For MOTR results, use xy or xyz distance for metric matching and slot alignment.')
    parser.add_argument('--motr-no-slot-align', action='store_true', help='For MOTR results, disable slot alignment before writing generate_results-compatible files.')
    parser.add_argument('--motr-plot-no-lines', action='store_true', default=True, help='For MOTR result plots, draw only points without connecting trajectory lines.')
    parser.add_argument('--stabilize-gt', dest='stabilize_gt', action='store_true', default=True, help='Stabilize GT identities by frame-to-frame GT position matching before tracking metrics.')
    parser.add_argument('--no-stabilize-gt', dest='stabilize_gt', action='store_false', help='Disable GT identity stabilization before tracking metrics.')
    parser.add_argument('--gt-stabilization-threshold', type=float, default=0.2, help='Max distance for GT identity stabilization. Defaults to 0.2 m.')
    parser.add_argument('--tracking-score-threshold', type=float, default=0.5, help='Score threshold for tracking metrics on DL/Kalman/DBSCAN result folders.')
    parser.add_argument('--kalman-pred-score-threshold', type=float, default=0.5, help='For non-MOTR Kalman results, minimum DL prediction score used as Kalman measurements.')
    parser.add_argument('--tracking-match-threshold', type=float, default=1.0, help='Distance threshold for tracking metrics on Kalman/DBSCAN result folders.')
    parser.add_argument('--tracking-match-dims', type=int, default=2, choices=(2, 3), help='Use xy or xyz distance for tracking metrics on Kalman/DBSCAN result folders.')
    parser.add_argument('-maf', '--kalman-max-active-filters', type=int, default=None, help='For non-MOTR Kalman plots/results, pass --max_active_filters to kalman.py.')
    parser.add_argument('--kalman-var-process-noise', type=float, default=0.14)
    parser.add_argument('--kalman-var-measurement-noise', type=float, default=3.0)
    parser.add_argument('--kalman-assignment-threshold', type=float, default=0.4)
    parser.add_argument('--kalman-max-miss', type=int, default=6)
    parser.add_argument('--kalman-filter-short-frames', type=int, default=0)
    parser.add_argument('--kalman-dim', type=int, choices=(2, 3), default=2)
    parser.add_argument('--evaluation-y-limit', type=float, default=4.9)
    args = parser.parse_args()
    path = args.input
    OUTPUT_SUFFIX = args.output_suffix
    DATASET_DIR = args.datasetdir
    use_motr = is_motr_result(path)
    if use_motr:
        MOTR_EVAL_ARGS.append('--save-preds')
        MOTR_EVAL_ARGS.extend(['--evaluation-y-limit', str(args.evaluation_y_limit)])
    if args.motr_window_batch_eval:
        MOTR_EVAL_ARGS.append('--window-batch-eval')
    if args.motr_context_replay:
        MOTR_EVAL_ARGS.append('--context-replay')
    if args.motr_score_threshold is not None:
        MOTR_EVAL_ARGS.extend(['--score-threshold', str(args.motr_score_threshold)])
    if args.motr_keep_track_score_threshold is not None:
        MOTR_EVAL_ARGS.extend(['--keep-track-score-threshold', str(args.motr_keep_track_score_threshold)])
    if args.motr_new_track_score_threshold is not None:
        MOTR_EVAL_ARGS.extend(['--new-track-score-threshold', str(args.motr_new_track_score_threshold)])
    if args.motr_max_inactive_steps is not None:
        MOTR_EVAL_ARGS.extend(['--max-inactive-steps', str(args.motr_max_inactive_steps)])
    if args.motr_promotion_suppression_radius is not None:
        MOTR_EVAL_ARGS.extend(['--promotion-suppression-radius', str(args.motr_promotion_suppression_radius)])
    if args.motr_match_threshold is not None:
        MOTR_EVAL_ARGS.extend(['--match-threshold', str(args.motr_match_threshold)])
    if args.motr_match_dims is not None:
        MOTR_EVAL_ARGS.extend(['--match-dims', str(args.motr_match_dims)])
    if args.stabilize_gt:
        MOTR_EVAL_ARGS.append('--stabilize-gt')
    if args.gt_stabilization_threshold is not None:
        MOTR_EVAL_ARGS.extend(['--gt-stabilization-threshold', str(args.gt_stabilization_threshold)])
    if args.motr_no_slot_align:
        MOTR_EVAL_ARGS.append('--no-slot-align')
    TRACKING_EVAL_ARGS.extend(['--tracking-score-threshold', str(args.tracking_score_threshold), '--tracking-match-threshold', str(args.tracking_match_threshold), '--tracking-match-dims', str(args.tracking_match_dims)])
    if args.stabilize_gt:
        TRACKING_EVAL_ARGS.append('--stabilize-gt')
    if args.gt_stabilization_threshold is not None:
        TRACKING_EVAL_ARGS.extend(['--gt-stabilization-threshold', str(args.gt_stabilization_threshold)])
    print('args.input ', args.input)
    print('use_motr_eval ', use_motr)
    print('motr_eval_args ', MOTR_EVAL_ARGS)
    print('output_suffix ', OUTPUT_SUFFIX)
    print('datasetdir ', DATASET_DIR)
    print('kalman_pred_score_threshold ', args.kalman_pred_score_threshold)
    print('kalman_max_active_filters ', args.kalman_max_active_filters)
    cmd_list = []
    if use_motr:
        plot_extra_args = ['--no-lines'] if args.motr_plot_no_lines else []
    else:
        plot_extra_args = ['--var_process_noise', str(args.kalman_var_process_noise), '--var_measurement_noise', str(args.kalman_var_measurement_noise), '--threshold', str(args.kalman_assignment_threshold), '--max_miss', str(args.kalman_max_miss), '--filter_short_frames', str(args.kalman_filter_short_frames), '--dim', str(args.kalman_dim), '--evaluation-y-limit', str(args.evaluation_y_limit), '--evaluation-y-limit-mode', 'ignore', '--kalman-reset-between-sequences', '--tracking-reset-between-sequences', '--tracking-score-threshold', str(args.tracking_score_threshold), '--tracking-match-threshold', str(args.tracking_match_threshold), '--tracking-match-dims', str(args.tracking_match_dims), '--kalman-pred-score-threshold', str(args.kalman_pred_score_threshold)]
        if args.stabilize_gt:
            plot_extra_args.append('--stabilize-gt')
        if args.gt_stabilization_threshold is not None:
            plot_extra_args.extend(['--gt-stabilization-threshold', str(args.gt_stabilization_threshold)])
        if args.kalman_max_active_filters is not None:
            plot_extra_args.extend(['-maf', str(args.kalman_max_active_filters)])
    plot_extra_args.extend(['--plot-x-limits', '-3', '3', '--plot-y-limits', '1', '6'])
    with open(os.path.join(path, 'config.yaml'), encoding='utf-8') as stream:
        run_config = yaml.safe_load(stream)
    for recording in run_config['DATASET']['test']:
        name = os.path.basename(recording.rstrip('/'))
        cmd_list.append(build_prediction_command(path, recording, name, use_motr))
        cmd_list.append([sys.executable, visualization_script(use_motr), '-i',
                         os.path.join(path, result_name(name)), '-no_p', '-s',
                         args.plot_format, '--frame_range', '600', '900'] + plot_extra_args)
    for cmd in cmd_list:
        if '-s' in cmd:
            save_arg_index = cmd.index('-s')
            if save_arg_index + 1 < len(cmd):
                cmd[save_arg_index + 1] = args.plot_format
        existing_prediction_path = prediction_result_path(cmd)
        if existing_prediction_path is not None and (not args.force_regenerate_preds) and prediction_outputs_exist(existing_prediction_path):
            print(f'Skipping existing prediction results: {existing_prediction_path}', flush=True)
            continue
        print('Running command:\n ' + ' '.join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Command failed with exit code {e.returncode}')
            print('Command:\n', ' '.join(cmd))
            raise SystemExit(e.returncode)
