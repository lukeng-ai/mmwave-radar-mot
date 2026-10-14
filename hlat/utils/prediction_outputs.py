import os
import numpy as np
_PREDS_OTHER_DISABLED_BY_DEFAULT_MODELS = {'PNTKPCONVTBD'}

def prediction_main_width(cfg):
    return int(cfg.REGRESSION_PARAMETER) * int(cfg.PNT_NUM_PEOPLE) + int(cfg.PNT_NUM_PEOPLE)

def should_save_preds_other(cfg):
    explicit_flag = getattr(cfg, 'SAVE_PREDS_OTHER_FLAG', None)
    if explicit_flag is not None:
        return bool(explicit_flag)
    model_name = str(getattr(cfg, 'MODEL', '')).upper()
    return model_name not in _PREDS_OTHER_DISABLED_BY_DEFAULT_MODELS

def save_preds_other_if_enabled(result_path, predict_tensor, cfg):
    start_index = prediction_main_width(cfg)
    if predict_tensor.shape[1] <= start_index:
        return False
    if not should_save_preds_other(cfg):
        print(f"Skipping preds_other.npy for {getattr(cfg, 'MODEL', 'model')}; set SAVE_PREDS_OTHER_FLAG=True to save auxiliary prediction data.")
        return False
    np.save(os.path.join(result_path, 'preds_other.npy'), predict_tensor[:, start_index:].cpu().numpy())
    return True
