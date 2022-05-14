import pdb
import torch
import random
import numpy as np

from Models.DiscreteBKI import DiscreteBKI

CLASS_COUNTS_REMAPPED = np.array([
    447156890,
    0,
    261005182,
    107172982,
    22852,
    224173,
    0,
    111345,
    2,
    479,
    10,
    554091,
    10626325,
    1588416,
    168764964,
    10944799,
    3502156,
    1493276,
    5798200,
    3395458,
], dtype=np.long)

# Intersection, union for one frame
def iou_one_frame(pred, target, n_classes=20):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection[cls] = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union[cls] = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection[cls]
    return intersection, union

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_model(model_name, grid_params, device):
    # Model parameters
    if model_name == "DiscreteBKI":
        B = 8
        model = DiscreteBKI(
            torch.tensor(grid_params['grid_size'], dtype=torch.long).to(device), # Grid size
            torch.tensor(grid_params['min_bound']).to(device), # Lower bound
            torch.tensor(grid_params['max_bound']).to(device), # Upper bound
            device=device
        )
        model.initialize_kernel()
        decayRate = 0.96
    else:
        print("Please choose either DiscreteBKI, Neural Blox, or Conv Occupancy. Thank you.")
        exit()
    return model, B, decayRate
