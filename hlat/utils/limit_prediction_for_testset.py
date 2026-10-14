import os
import argparse
import json
import numpy as np

def zero_frames(mask, preds_classification, preds_regression, targets_classification, targets_regression):
    preds_classification[mask, :] = 0
    preds_regression[mask, ...] = 0
    targets_classification[mask, :] = 0
    targets_regression[mask, ...] = 0

def apply_y_limit(preds_classification, preds_regression, targets_classification, targets_regression, y_limit, copy=True):
    arrays = (preds_classification, preds_regression, targets_classification, targets_regression)
    if copy:
        arrays = tuple((array.copy() for array in arrays))
    (preds_classification, preds_regression, targets_classification, targets_regression) = arrays
    if targets_classification.ndim != 2:
        raise ValueError(f'targets_classification must have shape [frames, people], got {targets_classification.shape}.')
    (num_frames, num_people) = targets_classification.shape
    if targets_regression.ndim == 2:
        if targets_regression.shape[0] != num_frames:
            raise ValueError(f'Target regression and classification frame counts differ: {targets_regression.shape[0]} != {num_frames}.')
        if targets_regression.shape[1] % num_people != 0:
            raise ValueError(f'Flattened targets_regression columns must be divisible by the number of people: {targets_regression.shape[1]} % {num_people} != 0.')
        targets_regression_view = targets_regression.reshape(num_frames, num_people, -1)
    elif targets_regression.ndim == 3:
        targets_regression_view = targets_regression
    else:
        raise ValueError(f'targets_regression must have shape [frames, people, dimensions] or [frames, people * dimensions], got {targets_regression.shape}.')
    if targets_regression_view.shape[-1] < 2:
        raise ValueError('targets_regression must contain at least x and y coordinates.')
    frames_over_limit = (targets_regression_view[:, :, 1] > float(y_limit)).any(axis=1)
    zero_frames(frames_over_limit, preds_classification, preds_regression, targets_classification, targets_regression)
    return (preds_classification, preds_regression, targets_classification, targets_regression, frames_over_limit)

def apply_rules_and_cut(preds_cls, preds_reg, t_cls, t_reg, limit, cut_no_label):
    """
    Apply Rule 1 (y-limit) and Rule 2 (cut_no_label) by REMOVING offending frames.
    Returns filtered arrays and a dict of cut counts.
    """
    F = preds_cls.shape[0]
    keep_mask = np.ones(F, dtype=bool)
    cut_counts = {'rule1_limit': 0, 'rule2_cut': 0}
    if limit is not None:
        y_vals = t_reg[:, :, 1]
        over_limit = (y_vals > float(limit)).any(axis=1)
        cut_counts['rule1_limit'] = int(over_limit.sum())
        keep_mask &= ~over_limit
    if cut_no_label is not None:
        try:
            cut_value = int(float(cut_no_label))
        except ValueError:
            raise ValueError(f'--cut_no_label must be numeric (1 or 2), got: {cut_no_label!r}')
        if cut_value == 2:
            is_11 = (t_cls == 1).all(axis=1)
            rule2_remove = ~is_11
            cut_counts['rule2_cut'] = int((rule2_remove & keep_mask).sum())
            keep_mask &= is_11
        elif cut_value == 1:
            pass
        else:
            raise ValueError('--cut_no_label must be 1 or 2 if provided.')
    return (preds_cls[keep_mask], preds_reg[keep_mask], t_cls[keep_mask], t_reg[keep_mask], cut_counts, keep_mask)

def split_into_three_idxs(F):
    """Return three contiguous index slices that split 0..F-1 into ~equal thirds."""
    idxs = np.arange(F)
    parts = np.array_split(idxs, 3)
    return parts

def filter_sequences_and_equalize(preds_cls, preds_reg, targets_cls, targets_reg, sequence_lengths, limit=4.9, cut_no_label=None, pred_track_ids=None, gt_track_ids=None):
    sequence_lengths = [int(length) for length in sequence_lengths]
    if any((length <= 0 for length in sequence_lengths)):
        raise ValueError('All sequence lengths must be positive.')
    if sum(sequence_lengths) != preds_cls.shape[0]:
        raise ValueError(f'Sequence lengths must sum to the number of frames: {sum(sequence_lengths)} != {preds_cls.shape[0]}.')
    part_results = []
    summary = {'total_frames_in': int(preds_cls.shape[0]), 'parts': []}
    start = 0
    for (part_index, length) in enumerate(sequence_lengths):
        end = start + length
        filtered = apply_rules_and_cut(preds_cls[start:end], preds_reg[start:end], targets_cls[start:end], targets_reg[start:end], limit, cut_no_label)
        (filtered_preds_cls, filtered_preds_reg, filtered_targets_cls, filtered_targets_reg, cut_counts, _) = filtered
        keep_mask = filtered[-1]
        filtered_track_ids = None if pred_track_ids is None else pred_track_ids[start:end][keep_mask]
        filtered_gt_track_ids = None if gt_track_ids is None else gt_track_ids[start:end][keep_mask]
        kept_len = int(filtered_preds_cls.shape[0])
        part_results.append((filtered_preds_cls, filtered_preds_reg, filtered_targets_cls, filtered_targets_reg, filtered_track_ids, filtered_gt_track_ids))
        summary['parts'].append({'part_index': part_index, 'orig_len': length, 'kept_len': kept_len, 'cut_rule1_limit': cut_counts['rule1_limit'], 'cut_rule2': cut_counts['rule2_cut']})
        start = end
    equalized_length = min((part[0].shape[0] for part in part_results))
    equalized_parts = []
    for part in part_results:
        equalized_parts.append(tuple((None if array is None else array[:equalized_length] for array in part)))
    summary['equalized_len_per_part'] = int(equalized_length)
    summary['total_frames_out'] = int(equalized_length * len(equalized_parts))
    result = (np.concatenate([part[0] for part in equalized_parts], axis=0), np.concatenate([part[1] for part in equalized_parts], axis=0), np.concatenate([part[2] for part in equalized_parts], axis=0), np.concatenate([part[3] for part in equalized_parts], axis=0), [int(equalized_length)] * len(equalized_parts), summary)
    if pred_track_ids is not None:
        result = (*result, np.concatenate([part[4] for part in equalized_parts], axis=0))
    if gt_track_ids is not None:
        result = (*result, np.concatenate([part[5] for part in equalized_parts], axis=0))
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='./data', help='Input folder containing the npy files.')
    parser.add_argument('-l', '--limit', default=4.9, help='If set, frames with any GT y-value > limit will be CUT (removed) within each part.')
    parser.add_argument('-c', '--cut_no_label', default=None, help='If 2: within each part, keep only frames with labels [1,1]; remove others. If 1: do not remove on this rule.')
    parser.add_argument('--outname', default=None, help='Optional name suffix for the output directory.')
    args = parser.parse_args()
    preds_classification = np.load(os.path.join(args.input, 'preds_classification.npy'))
    preds_regression = np.load(os.path.join(args.input, 'preds_regression.npy'))
    targets_classification = np.load(os.path.join(args.input, 'targets_classification.npy'))
    targets_regression = np.load(os.path.join(args.input, 'targets_regression.npy'))
    if targets_regression.ndim != 3 or targets_regression.shape[2] < 2:
        print('targets_regression must have at least 3 dimensions with size >=2 in the last dimension.')
        try:
            targets_regression = targets_regression.reshape(targets_regression.shape[0], 2, 6)
            print('Reshaped targets_regression to:', targets_regression.shape)
        except Exception as e:
            raise ValueError('Failed to reshape targets_regression.') from e
    if preds_regression.ndim != 3 or preds_regression.shape[2] < 2:
        print('preds_regression must have at least 3 dimensions with size >=2 in the last dimension.')
        try:
            preds_regression = preds_regression.reshape(preds_regression.shape[0], 2, 6)
            print('Reshaped preds_regression to:', preds_regression.shape)
        except Exception as e:
            raise ValueError('Failed to reshape preds_regression.') from e
    F = preds_classification.shape[0]
    assert preds_regression.shape[0] == F == targets_classification.shape[0] == targets_regression.shape[0], 'All arrays must share the same first dimension (frames).'
    parts = split_into_three_idxs(F)
    assert len(parts) == 3
    (final_preds_cls, final_preds_reg, final_t_cls, final_t_reg, _, summary) = filter_sequences_and_equalize(preds_classification, preds_regression, targets_classification, targets_regression, [len(part) for part in parts], limit=args.limit, cut_no_label=args.cut_no_label)
    min_len = summary['equalized_len_per_part']
    limit_tag = f'limit_{args.limit}' if args.limit is not None else 'nolimit'
    cut_tag = f'cut_{args.cut_no_label}' if args.cut_no_label is not None else 'nocut'
    base_name = args.outname if args.outname else f'parts3_filtered_{limit_tag}_{cut_tag}'
    out_dir = os.path.join(args.input, base_name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'preds_classification.npy'), final_preds_cls)
    np.save(os.path.join(out_dir, 'preds_regression.npy'), final_preds_reg)
    np.save(os.path.join(out_dir, 'targets_classification.npy'), final_t_cls)
    np.save(os.path.join(out_dir, 'targets_regression.npy'), final_t_reg)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('=== Per-part results (before equalization) ===')
    for part in summary['parts']:
        print(f"Part {part['part_index']}: orig={part['orig_len']}, kept={part['kept_len']}, cut_rule1={part['cut_rule1_limit']}, cut_rule2={part['cut_rule2']}")
    print(f'Equalized each part to length: {min_len}')
    print(f"Total frames after concat: {summary['total_frames_out']}")
    print(f'✅ Saved filtered arrays and summary to: {out_dir}')
