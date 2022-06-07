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
    74318204,    
    22488469,          
    8791,          
    18146,
    0,          
    0,          
    0,          
    10,          
    0,      
    165493,
    0,          
    893182,   
    29060462,     
    6046058,     
    355384,     
    269356,
    951300,     
    905264,  
    839469936
], dtype=np.long)

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_model(model_name, grid_params, device):
    # Model parameters
    if model_name == "DiscreteBKI":
        B = 20
        model = DiscreteBKI(
            torch.tensor(grid_params['grid_size'], dtype=torch.long).to(device), # Grid size
            torch.tensor(grid_params['min_bound']).to(device), # Lower bound
            torch.tensor(grid_params['max_bound']).to(device), # Upper bound
            device=device,
            datatype=torch.float32
        )
        model.initialize_kernel()
        decayRate = 0.96
    else:
        print("Please choose either DiscreteBKI, Neural Blox, or Conv Occupancy. Thank you.")
        exit()
    return model, B, decayRate