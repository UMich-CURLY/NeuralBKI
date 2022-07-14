import os
import pdb
import time
import json
import yaml
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
from Models.ConvBKI import *
from Data.Rellis3D import Rellis3dDataset
from Data.SemanticKitti import KittiDataset

MODEL_NAME = "ConvBKI_PerClass"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device == "cuda":
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
else:
  start = None
  end = None
print("device is ", device)

# import rospy
# from visualization_msgs.msg import *
# rospy.init_node('talker',disable_signals=True)
# map_pub = rospy.Publisher('SemMap', MarkerArray, queue_size=10)
# pred_pub = rospy.Publisher('SemPredMap', MarkerArray, queue_size=10)

model_params_file = os.path.join(os.getcwd(), "Config", MODEL_NAME + ".yaml")
with open(model_params_file, "r") as stream:
    try:
        model_params = yaml.safe_load(stream)
        dataset = model_params["dataset"]
    except yaml.YAMLError as exc:
        print(exc)

# CONSTANTS
SEED = model_params["seed"]
NUM_FRAMES = model_params["num_frames"]
MODEL_RUN_DIR = os.path.join("Models", "Runs", MODEL_NAME + "_" + dataset)
NUM_WORKERS = model_params["num_workers"]
FLOAT_TYPE = torch.float32
LABEL_TYPE = torch.uint8

if not os.path.exists(MODEL_RUN_DIR):
    os.makedirs(MODEL_RUN_DIR)
TRIAL_NUM = str(len(os.listdir(MODEL_RUN_DIR)))

# Model Parameters
data_params_file = os.path.join(os.getcwd(), "Config", dataset + ".yaml")
with open(data_params_file, "r") as stream:
    try:
        data_params = yaml.safe_load(stream)
        NUM_CLASSES = data_params["num_classes"]
        print("Num classes: ", NUM_CLASSES)
        # if dataset == "semantic_kitti": # kitti has remap so we hard code the frequencies
        #     class_frequencies = np.array([0, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
        #                         6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
        #                         2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
        #                         2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
        #                         2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05], dtype=np.long)
        # else:
        class_frequencies = np.asarray([data_params["class_counts"][i] for i in range(NUM_CLASSES)], dtype=np.compat.long)
        TRAIN_DIR = data_params["data_dir"]
    except yaml.YAMLError as exc:
        print(exc)

epsilon_w = 1e-5  # eps to avoid zero division
weights = torch.from_numpy( (1 / np.log(class_frequencies + epsilon_w) )).to(dtype=FLOAT_TYPE, device=device)

if dataset == "semantic_kitti":
    criterion = nn.NLLLoss(weight=weights, ignore_index=255) #jingyu edit: ignore 255 in the look up table
else:
    criterion = nn.NLLLoss(weight=weights)
# pdb.set_trace()
scenes = [ s for s in sorted(os.listdir(TRAIN_DIR)) if s.isdigit() ]

# Load model
lr = model_params["train"]["lr"]
BETA1 = model_params["train"]["BETA1"]
BETA2 = model_params["train"]["BETA2"]
decayRate = model_params["train"]["decayRate"]
B = model_params["train"]["B"]
EPOCH_NUM = model_params["train"]["num_epochs"]
model_params["device"] = device
model_params["num_classes"] = NUM_CLASSES
model_params["datatype"] = FLOAT_TYPE
model = get_model(MODEL_NAME, model_params=model_params)

if dataset == "rellis":
    train_ds = Rellis3dDataset(model_params["train"]["grid_params"], directory=TRAIN_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=False)
    dataloader_train = DataLoader(train_ds, batch_size=B, shuffle=True, collate_fn=train_ds.collate_fn, num_workers=NUM_WORKERS)

    val_ds  = Rellis3dDataset(model_params["train"]["grid_params"], directory=TRAIN_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=False, model_setting="val")
    dataloader_val = DataLoader(val_ds, batch_size=B, shuffle=True, collate_fn=val_ds.collate_fn, num_workers=NUM_WORKERS)

if dataset == "semantic_kitti":
    train_ds = KittiDataset(model_params["train"]["grid_params"], directory=TRAIN_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=False)
    dataloader_train = DataLoader(train_ds, batch_size=B, shuffle=True, collate_fn=train_ds.collate_fn, num_workers=NUM_WORKERS)

    val_ds  = KittiDataset(model_params["train"]["grid_params"], directory=TRAIN_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=False, split="valid")
    dataloader_val = DataLoader(val_ds, batch_size=B, shuffle=True, collate_fn=val_ds.collate_fn, num_workers=NUM_WORKERS)
    # pass # TODO

trial_dir = os.path.join(MODEL_RUN_DIR, "t"+TRIAL_NUM)
save_dir = os.path.join("Models", "Weights", MODEL_NAME + "_" + dataset, "t"+TRIAL_NUM)

if not os.path.exists(trial_dir):
    os.makedirs(trial_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

training_log = open(os.path.join(trial_dir, "training_log.txt"), "a")

writer = SummaryWriter(os.path.join(MODEL_RUN_DIR, "t"+TRIAL_NUM))

# Optimizer setup
setup_seed(SEED)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(BETA1, BETA2))
# optimizer = optim.SGD(model.parameters(), lr=lr)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=1e-4, verbose=True)

train_count = 0
min_bound_torch = torch.from_numpy(train_ds.min_bound).to(device=device)
grid_dims_torch = torch.from_numpy(train_ds.grid_dims).to(dtype=torch.int, device=device)
voxel_sizes_torch = torch.from_numpy(train_ds.voxel_sizes).to(device=device)


def semantic_loop(dataloader, epoch, train_count=None, training=False):
    num_correct = 0
    num_total = 0
    all_intersections = np.zeros(NUM_CLASSES)
    all_unions = np.zeros(NUM_CLASSES) + 1e-6  # SMOOTHING

    for points, points_labels, gt_labels in dataloader:
        batch_gt = torch.zeros((0, 1), device=device, dtype=LABEL_TYPE)
        batch_preds = torch.zeros((0, NUM_CLASSES), device=device, dtype=FLOAT_TYPE)

        optimizer.zero_grad()
        for b in range(len(points)):
            current_map = model.initialize_grid()

            pc_np = np.vstack(np.array(points[b]))
            labels_np = np.vstack(np.array(points_labels[b]))
            labeled_pc = np.hstack((pc_np, labels_np))

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
            training_log.write(f'{str(model.ell.data)}\n')
            training_log.flush()
            # print(model.ell)

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
        training_log.write(f'Epoch {epoch} out of {EPOCH_NUM}, complete \n')
        training_log.flush()


    if not training:
        all_intersections = all_intersections[all_unions > 0]
        all_unions = all_unions[all_unions > 0]
        print(f'Epoch Num: {epoch} ------ average val accuracy: {num_correct/num_total}')
        training_log.write(f'Epoch Num: {epoch} ------ average val accuracy: {num_correct/num_total}\n')
        print(f'Epoch Num: {epoch} ------ val miou: {np.mean(all_intersections / all_unions)}')
        training_log.write(f'Epoch Num: {epoch} ------ val miou: {np.mean(all_intersections / all_unions)}')
        training_log.flush()
        writer.add_scalar(MODEL_NAME + '/Accuracy/Val', num_correct/num_total, epoch)
        writer.add_scalar(MODEL_NAME + '/mIoU/Val', np.mean(all_intersections / all_unions), epoch)
    
    return model, train_count


for epoch in range(EPOCH_NUM):
    # Start with validation
    model.eval()
    with torch.no_grad():
        semantic_loop(dataloader_val, epoch, training=False)

    # Training
    model.train()
    idx = 0
    model, train_count = semantic_loop(dataloader_train, epoch, train_count=train_count, training=True)
training_log.close()
writer.close()