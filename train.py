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
from Data.utils import *
from Models.model_utils import *
from Models.ConvBKI import *
from Data.Rellis3D import Rellis3dDataset
from Data.SemanticKitti import KittiDataset

MODEL_NAME = "ConvBKI_PerClass_Compound"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is ", device)
print("Model is", MODEL_NAME)

model_params_file = os.path.join(os.getcwd(), "Config", MODEL_NAME + ".yaml")
with open(model_params_file, "r") as stream:
    try:
        model_params = yaml.safe_load(stream)
        dataset = model_params["dataset"]
        SAVE_NAME = model_params["save_dir"]
    except yaml.YAMLError as exc:
        print(exc)

# CONSTANTS
SEED = model_params["seed"]
DEBUG_MODE = model_params["debug_mode"]
NUM_FRAMES = model_params["num_frames"]
MODEL_RUN_DIR = os.path.join("Models", "Runs", SAVE_NAME)
NUM_WORKERS = model_params["num_workers"]
FLOAT_TYPE = torch.float32
LABEL_TYPE = torch.uint8

if not os.path.exists(MODEL_RUN_DIR):
    os.makedirs(MODEL_RUN_DIR)

# Data Parameters
data_params_file = os.path.join(os.getcwd(), "Config", dataset + ".yaml")
with open(data_params_file, "r") as stream:
    try:
        data_params = yaml.safe_load(stream)
        NUM_CLASSES = data_params["num_classes"]
        class_frequencies = np.asarray([data_params["class_counts"][i] for i in range(NUM_CLASSES)])
        TRAIN_DIR = data_params["data_dir"]
    except yaml.YAMLError as exc:
        print(exc)

epsilon_w = 1e-5  # eps to avoid zero division
weights = torch.from_numpy( (1 / np.log(class_frequencies + epsilon_w) )).to(dtype=FLOAT_TYPE, device=device)

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
    train_ds = Rellis3dDataset(model_params["train"]["grid_params"], directory=TRAIN_DIR, device=device,
                               num_frames=NUM_FRAMES, remap=True, use_aug=False)
    val_ds = Rellis3dDataset(model_params["train"]["grid_params"], directory=TRAIN_DIR, device=device,
                             num_frames=NUM_FRAMES, remap=True, use_aug=False, data_split="val")
if dataset == "semantic_kitti":
    # Save splits info
    train_ds = KittiDataset(model_params["train"]["grid_params"], directory=TRAIN_DIR, device=device,
                            num_frames=NUM_FRAMES, remap=True, use_aug=False)
    val_ds = KittiDataset(model_params["train"]["grid_params"], directory=TRAIN_DIR, device=device,
                          num_frames=NUM_FRAMES, remap=True, use_aug=False, data_split="val")

dataloader_train = DataLoader(train_ds, batch_size=B, shuffle=True, collate_fn=train_ds.collate_fn, num_workers=NUM_WORKERS)
dataloader_val = DataLoader(val_ds, batch_size=B, shuffle=False, collate_fn=val_ds.collate_fn, num_workers=NUM_WORKERS)

trial_dir = MODEL_RUN_DIR
save_dir = os.path.join("Models", "Weights", SAVE_NAME)
if not DEBUG_MODE:
    if os.path.exists(save_dir):
        print("Error: path already exists")
        exit()

if not os.path.exists(trial_dir):
    os.makedirs(trial_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

writer = SummaryWriter(MODEL_RUN_DIR)

# Optimizer setup
setup_seed(SEED)
if model_params["train"]["opt"] == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(BETA1, BETA2))
else:
    optimizer = optim.SGD(model.parameters(), lr=lr)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=1e-4, verbose=True)

train_count = 0
min_bound_torch = torch.from_numpy(train_ds.min_bound).to(device=device)
grid_dims_torch = torch.from_numpy(train_ds.grid_dims).to(dtype=torch.int, device=device)
voxel_sizes_torch = torch.from_numpy(train_ds.voxel_sizes).to(device=device)

if DEBUG_MODE:
    rospy.init_node('talker', anonymous=True)
    map_pub = rospy.Publisher('SemMap_global', MarkerArray, queue_size=10)


def semantic_loop(dataloader, epoch, train_count=None, training=False):
    num_correct = 0
    num_total = 0
    all_intersections = np.zeros(NUM_CLASSES)
    all_unions = np.zeros(NUM_CLASSES) + 1e-6  # SMOOTHING
    next_map = MarkerArray()

    for points, points_labels, gt_labels in dataloader:
        batch_gt = torch.zeros((0, 1), device=device, dtype=LABEL_TYPE)
        batch_preds = torch.zeros((0, NUM_CLASSES), device=device, dtype=FLOAT_TYPE)

        optimizer.zero_grad()
        for b in range(len(points)):
            current_map = model.initialize_grid()

            if model_params["train"]["remove_last"]:
                pc_np = np.vstack(np.array(points[b][:-1]))
                labels_np = np.vstack(np.array(points_labels[b][:-1]))
            else:
                pc_np = np.vstack(np.array(points[b]))
                labels_np = np.vstack(np.array(points_labels[b]))
            labeled_pc = np.hstack((pc_np, labels_np))

            if labeled_pc.shape[0] == 0:  # Zero padded
                print("Something is very wrong!")
                exit()

            labeled_pc_torch = torch.from_numpy(labeled_pc).to(device=device)
            preds = model(current_map, labeled_pc_torch)
            gt_sem_labels = torch.from_numpy(gt_labels[b]).to(device=device)

            if DEBUG_MODE:
                grid_params = model_params["test"]["grid_params"]
                colors = remap_colors(data_params["colors"])
                if rospy.is_shutdown():
                    exit("Closing Python")
                try:
                    next_map = publish_local_map(preds, model.centroids, grid_params, colors, next_map)
                    map_pub.publish(next_map)
                except:
                    exit("Publishing broke")

            last_pc_torch = torch.from_numpy(points[b][-1]).to(device=device)
            sem_preds = points_to_voxels_torch(preds, last_pc_torch,
                                               min_bound_torch, grid_dims_torch, voxel_sizes_torch)

            # Evaluate on last frame in scan (most recent one)
            sem_preds = sem_preds / torch.sum(sem_preds, dim=-1, keepdim=True)

            # Remove all that are 0 zero label
            # TODO change to use ignore list
            non_void_mask = gt_sem_labels[:, 0] != 0

            batch_gt = torch.vstack((batch_gt, gt_sem_labels[non_void_mask, :]))
            batch_preds = torch.vstack((batch_preds, sem_preds[non_void_mask, :]))

        batch_gt = batch_gt.reshape(-1)
        loss = criterion(torch.log(batch_preds), batch_gt.long())

        if training:
            loss.backward()
            print("H:", model.ell_h)
            print("Z:", model.ell_z)
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
            writer.add_scalar(SAVE_NAME + '/Loss/Train', loss.item(), train_count)
            writer.add_scalar(SAVE_NAME + '/Accuracy/Train', accuracy, train_count)
            writer.add_scalar(SAVE_NAME + '/mIoU/Train', np.mean(inter / union), train_count)

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
        writer.add_scalar(SAVE_NAME + '/Accuracy/Val', num_correct/num_total, epoch)
        writer.add_scalar(SAVE_NAME + '/mIoU/Val', np.mean(all_intersections / all_unions), epoch)

    return model, train_count


def save_filter(model, save_path):
    filters = model.get_filters()
    torch.save(filters, save_path)


for epoch in range(EPOCH_NUM):
    # Save filters before any training
    if not DEBUG_MODE:
        save_filter(model, os.path.join("Models", "Weights", SAVE_NAME, "filters" + str(epoch) + ".pt"))

    # Validation
    # model.eval()
    # with torch.no_grad():
    #     semantic_loop(dataloader_val, epoch, training=False)

    # Training
    model.train()
    idx = 0
    model, train_count = semantic_loop(dataloader_train, epoch, train_count=train_count, training=True)

writer.close()
