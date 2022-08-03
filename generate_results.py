# This file generates results for evaluation by loading semantic predictions from files.
# Not intended for use on-board robot.

import os
import pdb
import time
import json

import rospy
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
from Models.mapping_utils import *
from Data.SemanticKitti import KittiDataset

MODEL_NAME = "ConvBKI_Single"
print("Model is:", MODEL_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is ", device)

# Model Parameters
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
MAP_METHOD = model_params["map_method"]
LOAD_EPOCH = model_params["load_epoch"]
LOAD_DIR = model_params["save_dir"]
VISUALIZE = model_params["visualize"]
MEAS_RESULT = model_params["meas_result"]
FROM_CONT = model_params["from_continuous"]
TO_CONT = model_params["to_continuous"]
PRED_PATH = model_params["pred_path"]

# Data Parameters
data_params_file = os.path.join(os.getcwd(), "Config", dataset + ".yaml")
with open(data_params_file, "r") as stream:
    try:
        data_params = yaml.safe_load(stream)
        NUM_CLASSES = data_params["num_classes"]
        colors = remap_colors(data_params["colors"])
        DATA_DIR = data_params["data_dir"]
    except yaml.YAMLError as exc:
        print(exc)

# Load data set
if dataset == "rellis":
    test_ds = Rellis3dDataset(model_params["test"]["grid_params"], directory=DATA_DIR, device=device,
                              num_frames=NUM_FRAMES, remap=True, use_aug=False, data_split="test")
elif dataset == "semantic_kitti":
    if MEAS_RESULT:
        test_ds = KittiDataset(model_params["test"]["grid_params"], directory=DATA_DIR, device=device,
                               num_frames=NUM_FRAMES, remap=True, use_aug=False, data_split=model_params["result_split"],
                               from_continuous=FROM_CONT, to_continuous=TO_CONT, pred_path=PRED_PATH)
    else:
        test_ds = KittiDataset(model_params["test"]["grid_params"], directory=DATA_DIR, device=device,
                               num_frames=NUM_FRAMES, remap=True, use_aug=False, data_split="test",
                               from_continuous=FROM_CONT, to_continuous=TO_CONT, pred_path=PRED_PATH)
dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=NUM_WORKERS)


# Create map object
grid_params = model_params["test"]["grid_params"]
if MAP_METHOD == "local":
    map_object = LocalMap(
        torch.tensor([int(p) for p in grid_params['grid_size']], dtype=torch.long).to(device),  # Grid size
        torch.tensor(grid_params['min_bound']).to(device),  # Lower bound
        torch.tensor(grid_params['max_bound']).to(device),  # Upper bound
        torch.load(os.path.join("Models", "Weights", LOAD_DIR, "filters" + str(LOAD_EPOCH) + ".pt")), # Filters
        model_params["filter_size"], # Filter size
        num_classes=NUM_CLASSES, # Classes
        device=device # Device
    )
elif MAP_METHOD == "global":
    map_object = GlobalMap(
        torch.tensor([int(p) for p in grid_params['grid_size']], dtype=torch.long).to(device),  # Grid size
        torch.tensor(grid_params['min_bound']).to(device),  # Lower bound
        torch.tensor(grid_params['max_bound']).to(device),  # Upper bound
        torch.load(os.path.join("Models", "Weights", LOAD_DIR, "filters" + str(LOAD_EPOCH) + ".pt")), # Filters
        model_params["filter_size"], # Filter size
        num_classes=NUM_CLASSES, # Classes
        device=device # Device
    )

if VISUALIZE:
    rospy.init_node('talker', anonymous=True)
    map_pub = rospy.Publisher('SemMap_global', MarkerArray, queue_size=10)
    next_map = MarkerArray()

# Iteratively loop through each scan
current_scene = None
current_frame_id = None
total_class = torch.zeros(map_object.num_classes, device=device)
total_int_bki = torch.zeros(map_object.num_classes, device=device)
total_int_seg = torch.zeros(map_object.num_classes, device=device)
total_un_bki = torch.zeros(map_object.num_classes, device=device)
total_un_seg = torch.zeros(map_object.num_classes, device=device)
for idx in range(len(test_ds)):
    with torch.no_grad():
        # Load data
        pose, points, pred_labels, gt_labels, scene_id, frame_id = test_ds.get_test_item(idx, get_gt=MEAS_RESULT)

        # Reset if new subsequence
        if scene_id != current_scene or (frame_id - 1) != current_frame_id:
            map_object.reset_grid()
        # Update pose if not
        map_object.propagate(pose)

        # Add points to map
        labeled_pc = np.hstack((points, pred_labels))
        labeled_pc_torch = torch.from_numpy(labeled_pc).to(device=device)
        map_object.update_map(labeled_pc_torch)

        current_scene = scene_id
        current_frame_id = frame_id

        if VISUALIZE:
            if rospy.is_shutdown():
                exit("Closing Python")
            try:
                map = publish_voxels(map_object, grid_params['min_bound'], grid_params['max_bound'], grid_params['grid_size'], colors, next_map)
                map_pub.publish(map)
            except:
                exit("Publishing broke")

        if MEAS_RESULT:
            predictions, local_mask = map_object.label_points(points)
            pred_labels = torch.from_numpy(pred_labels).to(device)
            if pred_labels.shape[1] > 1:
                pred_labels = torch.argmax(pred_labels, dim=1)
            else:
                pred_labels = pred_labels.view(-1)
            gt_labels = torch.from_numpy(gt_labels).to(device).view(-1)
            # TODO: Mask here?
            gt_labels[~local_mask] = 0
            pred_labels[~local_mask] = 0
            for i in range(1, map_object.num_classes):
                gt_i = gt_labels == i
                pred_bki_i = predictions == i
                pred_seg_i = pred_labels == i

                total_class[i] += torch.sum(gt_i)
                total_int_bki[i] += torch.sum(gt_i & pred_bki_i)
                total_int_seg[i] += torch.sum(gt_i & pred_seg_i)
                total_un_bki[i] += torch.sum(gt_i | pred_bki_i)
                total_un_seg[i] += torch.sum(gt_i | pred_seg_i)

            if idx % 100 == 0:
                print(idx, len(test_ds))
print("BKI:", total_int_bki / total_un_bki * 100)
print("Seg:", total_int_seg / total_un_seg * 100)