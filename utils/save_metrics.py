import torch
import json
import numpy as np
import json
import torch

def save_metrics_dict_to_json(loss_dict, json_filename):
    """
    Saves a dictionary of values to a JSON file.
    Converts any tensors to Python-native types (e.g., floats) before saving.

    Args:
        loss_dict (dict): A dictionary where keys are variable names and values are lists of loss values.
                          Loss values can be either scalars or tensors.
        json_filename (str): The filename for the output JSON file.
    """
    converted_loss_dict = {}
    for (key, loss_values) in loss_dict.items():
        if isinstance(loss_values, list):
            converted_loss_dict[key] = [float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss) if isinstance(loss, (np.float64, np.float32, float)) else loss for loss in loss_values]
        else:
            converted_loss_dict[key] = float(loss_values.item()) if isinstance(loss_values, torch.Tensor) else float(loss_values) if isinstance(loss_values, (np.float64, np.float32, float)) else loss_values
    with open(json_filename, 'w') as json_file:
        json.dump(converted_loss_dict, json_file)
