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
import copy
from tqdm import tqdm
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
from Data.KittiOdometry import KittiOdomDataset
import time

MODEL_NAME = "ConvBKI_Single"
# MODEL_NAME = "ConvBKI_Single_02_odom"

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
GEN_PREDS = model_params["gen_preds"]
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
        ignore_labels = data_params["ignore_labels"]

    except yaml.YAMLError as exc:
        print(exc)

print("Visualize Prediciton:", VISUALIZE)
print("Measure Result:", MEAS_RESULT)
print("Generate Prediction:", GEN_PREDS)
print("")
# Exit if measure result on test set
if MEAS_RESULT and model_params["result_split"] == "test":
    print("Error! Measure result can only be ran on train/val sets, test set does not have ground truth labels.")
    exit()


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
                               num_frames=NUM_FRAMES, remap=True, use_aug=False, data_split=model_params["result_split"],
                               from_continuous=FROM_CONT, to_continuous=TO_CONT, pred_path=PRED_PATH)
elif dataset == "kitti_odometry":
    if MEAS_RESULT:
        test_ds = KittiOdomDataset(model_params["train"]["grid_params"], directory=DATA_DIR, device=device,
                                num_frames=NUM_FRAMES, remap=False, use_aug=False, data_split=model_params["result_split"], from_continuous=FROM_CONT,
                                to_continuous=TO_CONT)
    else:
        test_ds = KittiOdomDataset(model_params["train"]["grid_params"], directory=DATA_DIR, device=device,
                            num_frames=NUM_FRAMES, remap=False, use_aug=False, data_split=model_params["result_split"], from_continuous=FROM_CONT,
                            to_continuous=TO_CONT)
                            
dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

# Create map object
grid_params = model_params["test"]["grid_params"]

map_object = GlobalMap(
    torch.tensor([int(p) for p in grid_params['grid_size']], dtype=torch.long).to(device),  # Grid size
    torch.tensor(grid_params['min_bound']).to(device),  # Lower bound
    torch.tensor(grid_params['max_bound']).to(device),  # Upper bound
    torch.load(os.path.join("Models", "Weights", LOAD_DIR, "filters" + str(LOAD_EPOCH) + ".pt")), # Filters
    model_params["filter_size"], # Filter size
    num_classes=NUM_CLASSES,
    ignore_labels = ignore_labels, # Classes
    device=device # Device
)

if VISUALIZE:
    rospy.init_node('talker', anonymous=True)
    map_pub = rospy.Publisher('SemMap_global', MarkerArray, queue_size=10)
    next_map = MarkerArray()

if GEN_PREDS:
    if not os.path.exists(MODEL_NAME):
        os.mkdir(MODEL_NAME)

# Iteratively loop through each scan
current_scene = None
current_frame_id = None
seq_dir = None
frame_num = 0
total_class = torch.zeros(map_object.num_classes, device=device)
total_int_bki = torch.zeros(map_object.num_classes, device=device)
total_int_seg = torch.zeros(map_object.num_classes, device=device)
total_un_bki = torch.zeros(map_object.num_classes, device=device)
total_un_seg = torch.zeros(map_object.num_classes, device=device)

total_t = 0.0
for idx in tqdm(range(len(test_ds))):
    with torch.no_grad():
        # Load data
        get_gt = model_params["result_split"] == "train" or model_params["result_split"] == "val" 
        pose, points, pred_labels, gt_labels, scene_id, frame_id = test_ds.get_test_item(idx, get_gt=get_gt)
        
        if VISUALIZE and MEAS_RESULT:
            if dataset == "semantic_kitti":
                not_void = (gt_labels != 0)[:, 0]
                points = points[not_void, :]
                pred_labels = pred_labels[not_void, :]
                gt_labels = gt_labels[not_void, :]

        if GEN_PREDS and seq_dir is None:
            seq_dir = os.path.join(MODEL_NAME, "sequences", str(scene_id).zfill(2), "predictions")

        # Reset if new subsequence
        if scene_id != current_scene or (frame_id - 1) != current_frame_id:
            map_object.reset_grid()
            if GEN_PREDS:
                seq_dir = os.path.join(MODEL_NAME, "sequences", str(scene_id).zfill(2), "predictions")
                frame_num = 0
                if not os.path.exists(seq_dir):
                    os.makedirs(seq_dir)
        # Update pose if not
        start_t = time.time()
        map_object.propagate(pose)

        # Add points to map
        labeled_pc = np.hstack((points, pred_labels))
        labeled_pc_torch = torch.from_numpy(labeled_pc).to(device=device, non_blocking=True)
        map_object.update_map(labeled_pc_torch)
        total_t += time.time() - start_t

        current_scene = scene_id
        current_frame_id = frame_id

        if VISUALIZE:
            if rospy.is_shutdown():
                exit("Closing Python")
            try:
                if MAP_METHOD == "global" or MAP_METHOD == "local":
                    map = publish_voxels(map_object, grid_params['min_bound'], grid_params['max_bound'], grid_params['grid_size'], colors, next_map)
                    map_pub.publish(map)
                elif MAP_METHOD == "local":
                    map = publish_local_map(map_object.local_map, map_object.centroids, grid_params, colors, next_map)
                    map_pub.publish(map)
            except:
                exit("Publishing broke")

        if MEAS_RESULT:
            if dataset == "semantic_kitti":
                # Filter out ignore labels
                non_ignore_mask = (gt_labels != ignore_labels[0])[:, 0]
                points = points[non_ignore_mask, :]
                gt_labels = gt_labels[non_ignore_mask, :]
                pred_labels = pred_labels[non_ignore_mask, :]
                # Make predictions and measure
                predictions, local_mask = map_object.label_points(points)
                pred_labels = torch.from_numpy(pred_labels).to(device, non_blocking=True)
                if pred_labels.shape[1] > 1:
                    pred_labels = torch.argmax(pred_labels, dim=1)
                else:
                    pred_labels = pred_labels.view(-1)
                gt_labels = torch.from_numpy(gt_labels).to(device, non_blocking=True).view(-1)
                # TODO: Change this line if needed. Maps outside local mask to segmentation labels.
                predictions_temp = pred_labels.detach().clone().to(predictions.dtype)
                predictions_temp[local_mask] = predictions[local_mask]
                predictions = predictions_temp

                for i in range(1, map_object.num_classes):
                    gt_i = gt_labels == i
                    pred_bki_i = predictions == i
                    pred_seg_i = pred_labels == i

                    total_class[i] += torch.sum(gt_i)
                    total_int_bki[i] += torch.sum(gt_i & pred_bki_i)
                    total_int_seg[i] += torch.sum(gt_i & pred_seg_i)
                    total_un_bki[i] += torch.sum(gt_i | pred_bki_i)
                    total_un_seg[i] += torch.sum(gt_i | pred_seg_i)
                
                if idx % 100 == 0 and not GEN_PREDS:
                    print(idx, len(test_ds))
                    print("BKI:", total_int_bki / total_un_bki * 100)
                    print("Seg:", total_int_seg / total_un_seg * 100)

            if dataset == "kitti_odometry":
                dists = np.linalg.norm(points, axis=1)
                in_range = dists < 40
                points = points[in_range, :]
                gt_labels = gt_labels[in_range]
                pred_labels = pred_labels[in_range]
            
                predictions, local_mask = map_object.label_points(points)
                
                pred_labels = torch.from_numpy(pred_labels).to(device, non_blocking=True)
                if pred_labels.shape[1] > 1:
                    pred_labels = torch.argmax(pred_labels, dim=1)
                else:
                    pred_labels = pred_labels.view(-1)
                gt_labels = torch.from_numpy(gt_labels).to(device, non_blocking=True).view(-1)

                # TODO: Mask here?
                gt_labels[~local_mask] = ignore_labels[0]
                pred_labels[~local_mask] = ignore_labels[0]

                for i in range(map_object.num_classes):
                    gt_i = gt_labels == i
                    pred_bki_i = predictions == i
                    pred_seg_i = pred_labels == i

                    total_class[i] += torch.sum(gt_i)
                    total_int_bki[i] += torch.sum(gt_i & pred_bki_i)
                    total_int_seg[i] += torch.sum(gt_i & pred_seg_i)
                    total_un_bki[i] += torch.sum(gt_i | pred_bki_i)
                    total_un_seg[i] += torch.sum(gt_i | pred_seg_i)


        if GEN_PREDS:
            frame_file = os.path.join(seq_dir, str(frame_num).zfill(6) + ".label")
            # Make predictions
            predictions, local_mask = map_object.label_points(points)
            if MEAS_RESULT:
                pred_labels = torch.unsqueeze(pred_labels, dim=-1)
                if pred_labels.shape[1] > 1:
                    pred_labels = torch.argmax(pred_labels, dim=1)  
                else:
                    pred_labels = pred_labels.view(-1)
            else:
                pred_labels = torch.from_numpy(pred_labels).to(device)
                if pred_labels.shape[1] > 1:
                    pred_labels = torch.argmax(pred_labels, dim=1)  
                else:
                    pred_labels = pred_labels.view(-1)
            
            # Maps outside local mask to segmentation labels.
            predictions_temp = pred_labels.detach().clone().to(predictions.dtype)
            predictions_temp[local_mask] = predictions[local_mask]
            predictions = predictions_temp.view(-1).detach().cpu().numpy().astype(np.uint32)
            # Save
            predictions.tofile(frame_file)

    frame_num += 1


if MEAS_RESULT:
    print("Final results:")
    if dataset == "kitti_odometry":
        bki_result = (total_int_bki / total_un_bki * 100).detach().cpu().numpy()
        seg_result = (total_int_seg / total_un_seg * 100).detach().cpu().numpy()
        bki_result_t = copy.deepcopy(bki_result)
        seg_result_t = copy.deepcopy(seg_result)
        Shift = [0, 1, 2, 3, 4, 7, 5, 8, 9, 6, 10]
        for i, label in enumerate(Shift):        
            bki_result[label] = bki_result_t[i]
            seg_result[label] = seg_result_t[i]
        print("BKI:")
        for i in range(bki_result.shape[0]-3):
            print(bki_result[i])
        print("Seg:")
        for i in range(seg_result.shape[0]-3):
            print(seg_result[i])
    else:
        print("Seg:")
        for i in range(NUM_CLASSES):
            print((total_int_seg[i] / total_un_seg[i] * 100).item())
        print("BKI:")
        for i in range(NUM_CLASSES):
            print((total_int_bki[i] / total_un_bki[i] * 100).item())
