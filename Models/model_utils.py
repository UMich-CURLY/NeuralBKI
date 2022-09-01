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


def measure_inf_time(model, inputs, reps=300):
    print(inputs.shape)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((reps, 1))
    model.eval()
    with torch.no_grad():
        current_map = model.initialize_grid()
        for rep in range(reps):
            starter.record()
            _ = model(current_map, inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / reps
    std_syn = np.std(timings)
    print(mean_syn)


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