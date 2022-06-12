import os
import pdb
import time
import json

from torch import batch_norm_update_stats, gt
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np

# Torch imports
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Custom Imports
from Benchmarks.eval_utils import iou_one_frame
from Data.utils import *
from Models.model_utils import *
from Models.DiscreteBKI import *
from Models.FocalLoss import FocalLoss
from Data.dataset import Rellis3dDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device == "cuda":
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
else:
  start = None
  end = None
print("device is ", device)

import rospy
from visualization_msgs.msg import *
rospy.init_node('talker',disable_signals=True)
map_pub = rospy.Publisher('SemMap', MarkerArray, queue_size=10)
pred_pub = rospy.Publisher('SemPredMap', MarkerArray, queue_size=10)

home_dir = os.path.expanduser('~')
dataset_loc = os.path.join(home_dir, "Data/Rellis-3D")

#CONSTANTS
SEED = 42
NUM_CLASSES = colors.shape[0]
DATASET_DIR = dataset_loc
NUM_FRAMES = 16
MODEL_NAME = "DiscreteBKI"
model_name = MODEL_NAME + "_" + str(NUM_CLASSES)
VISUALIZE_OUTPUT = True

MODEL_RUN_DIR = os.path.join("Models", "Runs", model_name)
if not os.path.exists(MODEL_RUN_DIR):
    os.makedirs(MODEL_RUN_DIR)
TRIAL_NUM = str(len(os.listdir(MODEL_RUN_DIR)))
NUM_WORKERS = 8
EPOCH_NUM = 500
FLOAT_TYPE = torch.float32
LABEL_TYPE = torch.uint8


#Model Parameters
class_frequencies = CLASS_COUNTS_REMAPPED
epsilon_w = 1e-5  # eps to avoid zero division
weights = torch.from_numpy( \
    (1 / np.log(class_frequencies + epsilon_w) )
).to(dtype=FLOAT_TYPE, device=device)

criterion = nn.NLLLoss(weight=weights)
# pdb.set_trace()
scenes = [ s for s in sorted(os.listdir(DATASET_DIR)) if s.isdigit() ]
model_params_file = os.path.join(DATASET_DIR, scenes[-1], 'params.json')
with open(model_params_file) as f:
    grid_params = json.load(f)
    grid_params['grid_size'] = [ int(p) for p in grid_params['grid_size'] ]

# Load model
model, B, decayRate = get_model(MODEL_NAME, grid_params=grid_params, device=device)

rellis_ds = Rellis3dDataset(directory=DATASET_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=False, model_setting="test")
dataloader = DataLoader(rellis_ds, batch_size=B, shuffle=True, collate_fn=rellis_ds.collate_fn, num_workers=NUM_WORKERS)

trial_dir = os.path.join(MODEL_RUN_DIR, "t"+TRIAL_NUM)
save_dir = os.path.join("Models", "Weights", model_name, "t"+TRIAL_NUM)

if not os.path.exists(trial_dir):
    os.makedirs(trial_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

writer = SummaryWriter(os.path.join(MODEL_RUN_DIR, "t"+TRIAL_NUM))

# Optimizer setup
setup_seed(SEED)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(BETA1, BETA2))
# optim.SGD(model.parameters(), lr=lr, momentum=0.9)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
# torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
#     T_max=100, eta_min=1e-4, verbose=True)

train_count = 0
min_bound_torch = torch.from_numpy(rellis_ds.min_bound).to(device=device)
grid_dims_torch = torch.from_numpy(rellis_ds.grid_dims).to(dtype=torch.int, device=device)
print(rellis_ds.voxel_sizes)

voxel_sizes_torch = torch.from_numpy(rellis_ds.voxel_sizes).to(device=device)

if VISUALIZE_OUTPUT:
    with torch.no_grad():
        for idx in range(len(rellis_ds)):
            

