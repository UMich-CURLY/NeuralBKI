import pdb
import torch
import random
import numpy as np
from torch import empty
from torch import long

from Models.DiscreteBKI import DiscreteBKI

CLASS_COUNTS_REMAPPED = np.array([
    0,         # Marked as zero because void is filtered out 
    0,   
    257352935,    
    64627941,          
    16752,          
    224016,
    0,          
    0,          
    0,          
    0,          
    0,      
    539944,
    0,          
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
        B = 40
        model = DiscreteBKI(
            torch.tensor(grid_params['grid_size'], dtype=torch.long).to(device), # Grid size
            torch.tensor(grid_params['min_bound']).to(device), # Lower bound
            torch.tensor(grid_params['max_bound']).to(device), # Upper bound
            filter_size=5,
            device=device,
            datatype=torch.float32
        )
        model.initialize_kernel()
        decayRate = 0.96
    else:
        print("Please choose either DiscreteBKI, Neural Blox, or Conv Occupancy. Thank you.")
        exit()
    return model, B, decayRate