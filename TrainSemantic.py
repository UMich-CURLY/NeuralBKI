import os
import pdb
import time
import json
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
from Models.DiscreteBKI_Kernel import *
from Models.FocalLoss import FocalLoss
from Data.SemanticSegmentation import Rellis3dDataset


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
TRAIN_DIR = dataset_loc
NUM_FRAMES = 3
MODEL_NAME = "DiscreteBKI_Kernel"
model_name = MODEL_NAME + "_" + str(NUM_CLASSES)

MODEL_RUN_DIR = os.path.join("Models", "Runs", model_name)
if not os.path.exists(MODEL_RUN_DIR):
    os.makedirs(MODEL_RUN_DIR)
TRIAL_NUM = str(len(os.listdir(MODEL_RUN_DIR)))
NUM_WORKERS = 16
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
scenes = [ s for s in sorted(os.listdir(TRAIN_DIR)) if s.isdigit() ]
model_params_file = os.path.join(TRAIN_DIR, scenes[-1], 'params.json')
with open(model_params_file) as f:
    grid_params = json.load(f)
    grid_params['grid_size'] = [ int(p) for p in grid_params['grid_size'] ]

# Load model
lr = 7e-3
BETA1 = 0.9
BETA2 = 0.999
model, B, decayRate = get_model(MODEL_NAME, grid_params=grid_params, device=device)

rellis_ds = Rellis3dDataset(directory=TRAIN_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=False)
dataloader_train = DataLoader(rellis_ds, batch_size=B, shuffle=True, collate_fn=rellis_ds.collate_fn, num_workers=NUM_WORKERS)

rellis_ds_val  = Rellis3dDataset(directory=TRAIN_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=False, model_setting="val")
dataloader_val = DataLoader(rellis_ds_val, batch_size=B, shuffle=True, collate_fn=rellis_ds_val.collate_fn, num_workers=NUM_WORKERS)

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
# optimizer = optim.SGD(model.parameters(), lr=lr)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=1e-4, verbose=True)

train_count = 0
min_bound_torch = torch.from_numpy(rellis_ds.min_bound).to(device=device)
grid_dims_torch = torch.from_numpy(rellis_ds.grid_dims).to(dtype=torch.int, device=device)
print(rellis_ds.voxel_sizes)
voxel_sizes_torch = torch.from_numpy(rellis_ds.voxel_sizes).to(device=device)


def semantic_loop(dataloader, epoch, train_count=None, training=False):
    num_correct = 0
    num_total = 0
    all_intersections = np.zeros(NUM_CLASSES)
    all_unions = np.zeros(NUM_CLASSES) + 1e-6  # SMOOTHING

    salsa_correct = 0
    salsa_intersections = np.zeros(NUM_CLASSES)
    salsa_unions = np.zeros(NUM_CLASSES) + 1e-6  # SMOOTHING

    for points, points_labels, gt_labels in dataloader:
        batch_gt = torch.zeros((0, 1), device=device, dtype=LABEL_TYPE)
        batch_preds = torch.zeros((0, NUM_CLASSES), device=device, dtype=FLOAT_TYPE)

        optimizer.zero_grad()
        for b in range(len(points)):
            current_map = model.initialize_grid()

            pc_np = np.vstack(np.array(points[b]))
            # TEST
            # labels_np = np.vstack(np.array(gt_labels[b]))
            labels_np = np.vstack(np.array(points_labels[b]))
            labeled_pc = np.hstack((pc_np, labels_np))
            # print(np.sum(np.array(points_labels[b])[0, :, 0] == np.array(gt_labels[b])[:, 0]) / labeled_pc.shape[0])

            if labeled_pc.shape[0] == 0:  # Zero padded
                print("Something is very wrong!")
                exit()

            labeled_pc_torch = torch.from_numpy(labeled_pc).to(device=device)
            preds = model(current_map, labeled_pc_torch)
            gt_sem_labels = torch.from_numpy(gt_labels[b]).to(device=device)

            last_pc_torch = torch.from_numpy(points[b][-1]).to(device=device)
            sem_preds = points_to_voxels_torch(preds, last_pc_torch,
                                               min_bound_torch, grid_dims_torch, voxel_sizes_torch)

            # Evaluate on last frame in scan (most recent one)
            sem_preds = sem_preds / torch.sum(sem_preds, dim=-1, keepdim=True)

            # Remove all that are 0 zero label
            non_void_mask = gt_sem_labels[:, 0] != 0

            batch_gt = torch.vstack((batch_gt, gt_sem_labels[non_void_mask, :]))
            batch_preds = torch.vstack((batch_preds, sem_preds[non_void_mask, :]))

        batch_gt = batch_gt.reshape(-1)
        loss = criterion(torch.log(batch_preds), batch_gt.long())

        if training:
            loss.backward()
            optimizer.step()

        # Accuracy
        with torch.no_grad():
            # Softmax on expectation
            max_batch_preds = torch.argmax(batch_preds, dim=-1)
            preds_masked = max_batch_preds.cpu().numpy()
            voxels_np = batch_gt.detach().cpu().numpy()
            num_correct += np.sum(preds_masked == voxels_np)
            num_total += voxels_np.shape[0]
            accuracy = np.sum(preds_masked == voxels_np) / voxels_np.shape[0]

            inter, union = iou_one_frame(max_batch_preds, batch_gt, n_classes=NUM_CLASSES)
            union += 1e-6
            all_intersections += inter
            all_unions += union

        # Record
        if training:
            writer.add_scalar(MODEL_NAME + '/Loss/Train', loss.item(), train_count)
            writer.add_scalar(MODEL_NAME + '/Accuracy/Train', accuracy, train_count)
            writer.add_scalar(MODEL_NAME + '/mIoU/Train', np.mean(inter / union), train_count)

            train_count += len(points)

    # Save model, decrease learning rate
    if training:
        my_lr_scheduler.step()
        print("Epoch ", epoch, " out of ", EPOCH_NUM, " complete.")

    if not training:
        all_intersections = all_intersections[all_unions > 0]
        all_unions = all_unions[all_unions > 0]
        print(f'Epoch Num: {epoch} ------ average val accuracy: {num_correct/num_total}')
        print(f'Epoch Num: {epoch} ------ val miou: {np.mean(all_intersections / all_unions)}')
        writer.add_scalar(MODEL_NAME + '/Accuracy/Val', num_correct/num_total, epoch)
        writer.add_scalar(MODEL_NAME + '/mIoU/Val', np.mean(all_intersections / all_unions), epoch)

    return model, train_count


for epoch in range(EPOCH_NUM):
    # Start with validation
    # model.eval()
    # with torch.no_grad():
    #     semantic_loop(dataloader_val, epoch, training=False)

    # Training
    model.train()
    idx = 0
    model, train_count = semantic_loop(dataloader_train, epoch, train_count=train_count, training=True)

writer.close()