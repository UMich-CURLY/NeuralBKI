import pdb
import torch
import random
import numpy as np
from torch import empty
from torch import long

from Models.DiscreteBKI import DiscreteBKI
from Models.DiscreteBKI_Kernel import DiscreteBKI_Kernel

CLASS_COUNTS_REMAPPED = np.array([
    0,         # Marked as zero because void is filtered out 
    0,   
    257352935,    
    64627941,          
    16752,          
    224016,
    0,          
    75812,          
    0,          
    0,          
    0,      
    539944,
    10538966,          
    1543817,   
    158171883,     
    9730727,     
    3474123,     
    1478073,
    5743794,     
    3345519,  
    0
], dtype=np.long)

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
            filter_size=9,
            device=device,
            datatype=torch.float32,
            kernel="random"
        )
        decayRate = 0.96
    elif model_name == "DiscreteBKI_Kernel":
        B = 1
        model = DiscreteBKI_Kernel(
            torch.tensor(grid_params['grid_size'], dtype=torch.long).to(device), # Grid size
            torch.tensor(grid_params['min_bound']).to(device), # Lower bound
            torch.tensor(grid_params['max_bound']).to(device), # Upper bound
            filter_size=15,
            device=device,
            datatype=torch.float32,
            kernel="sparse",
            max_dist=0.3,
            per_class=True
        )
        decayRate = 0.96
    elif model_name == "DiscreteBKI_SSC":
        B = 16
        model = DiscreteBKI(
            torch.tensor(grid_params['grid_size'], dtype=torch.long).to(device), # Grid size
            torch.tensor(grid_params['min_bound']).to(device), # Lower bound
            torch.tensor(grid_params['max_bound']).to(device), # Upper bound
            filter_size=3,
            device=device,
            datatype=torch.float32
        )
        decayRate = 0.96
    else:
        print("Please choose either DiscreteBKI, Neural Blox, or Conv Occupancy. Thank you.")
        exit()
    return model, B, decayRate