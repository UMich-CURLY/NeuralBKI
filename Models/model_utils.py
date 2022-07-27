import pdb
import torch
import random
import numpy as np
from torch import empty
from torch import long

from Models.ConvBKI import ConvBKI


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(model_name, model_params):
    # Model parameters
    grid_params = model_params["train"]["grid_params"]
    device = model_params["device"]

    try:
        model = ConvBKI(
            torch.tensor([int(p) for p in grid_params['grid_size']], dtype=torch.long).to(device),  # Grid size
            torch.tensor(grid_params['min_bound']).to(device),  # Lower bound
            torch.tensor(grid_params['max_bound']).to(device),  # Upper bound
            num_classes=model_params["num_classes"],
            filter_size=model_params["filter_size"],
            device=device,
            datatype=model_params["datatype"],
            kernel=model_params["kernel"],
            max_dist=model_params["ell"],
            per_class=model_params["per_class"],
            compound=model_params["compound"]
        )
    except:
        exit("Invalid config file.")
    return model