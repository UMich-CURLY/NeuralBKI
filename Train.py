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
from model_utils import *
from Models.DiscreteBKI import *
from Models.FocalLoss import FocalLoss
from Data.dataset import Rellis3dDataset, ray_trace_batch


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
NUM_FRAMES = 5
MODEL_NAME = "DiscreteBKI"
model_name = MODEL_NAME + "_" + str(NUM_CLASSES)
USE_FREE_SPACE = False

MODEL_RUN_DIR = os.path.join("Models", "Runs", model_name)
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

criterion = nn.CrossEntropyLoss(weight=weights) #FocalLoss(gamma=2, alpha=weights, device=device)
    # nn.CrossEntropyLoss(weight=weights)
# pdb.set_trace()
scenes = [ s for s in sorted(os.listdir(TRAIN_DIR)) if s.isdigit() ]
model_params_file = os.path.join(TRAIN_DIR, scenes[-1], 'params.json')
with open(model_params_file) as f:
    grid_params = json.load(f)
    grid_params['grid_size'] = [ int(p) for p in grid_params['grid_size'] ]

# Load model
lr = 1e-1
BETA1 = 0.9
BETA2 = 0.999
model, B, decayRate = get_model(MODEL_NAME, grid_params=grid_params, device=device)

rellis_ds = Rellis3dDataset(directory=TRAIN_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=False, use_voxels=USE_FREE_SPACE, use_gt=False)
dataloader = DataLoader(rellis_ds, batch_size=B, shuffle=False, collate_fn=rellis_ds.collate_fn, num_workers=NUM_WORKERS)

rellis_ds_val  = Rellis3dDataset(directory=TRAIN_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=False, use_voxels=USE_FREE_SPACE, use_gt=False, model_setting="val")
dataloader_val = DataLoader(rellis_ds_val, batch_size=B, shuffle=False, collate_fn=rellis_ds_val.collate_fn, num_workers=NUM_WORKERS)

trial_dir = os.path.join(MODEL_RUN_DIR, "t"+TRIAL_NUM)
save_dir = os.path.join("Models", "Weights", model_name, "t"+TRIAL_NUM)

if not os.path.exists(trial_dir):
    os.mkdir(trial_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

writer = SummaryWriter(os.path.join(MODEL_RUN_DIR, "t"+TRIAL_NUM))

# Optimizer setup
setup_seed(SEED)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(BETA1, BETA2))
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
# torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
#     T_max=100, eta_min=1e-4, verbose=True)

train_count = 0
min_bound_torch = torch.from_numpy(rellis_ds.min_bound).to(device=device)
grid_dims_torch = torch.from_numpy(rellis_ds.grid_dims).to(dtype=torch.int, device=device)
voxel_sizes_torch = torch.from_numpy(rellis_ds.voxel_sizes).to(device=device)
current_map = model.initialize_grid()

for epoch in range(EPOCH_NUM):
    # Training
    model.train()
    idx = 0
    for points, points_labels, gt_labels, invalid_voxels, occupied_voxels in dataloader:
        batch_labels = torch.zeros((0, 1), device=device, dtype=LABEL_TYPE)
        batch_preds = torch.zeros((0, NUM_CLASSES), device=device, dtype=FLOAT_TYPE)

        optimizer.zero_grad()
        for b in range(len(points)):
            pc_np = np.vstack(np.array(points[b]))
            labels_np = np.vstack(np.array(points_labels[b]))
            labeled_pc = np.hstack((pc_np, labels_np))

            if labeled_pc.shape[0]==0: # Zero padded
                print("continue")
                continue

            # Train over ssc with free space, otherwise semantic segmentation on pc
            if not USE_FREE_SPACE:
                labeled_pc_torch = torch.from_numpy(labeled_pc).to(device=device)
                preds = model(current_map, labeled_pc_torch)
                sem_labels = torch.from_numpy(gt_labels[b]).to(device=device)

                last_pc_torch = torch.from_numpy(points[b][-1]).to(device=device)
                sem_labels = points_to_voxels_torch(sem_labels, last_pc_torch,
                    min_bound_torch, grid_dims_torch, voxel_sizes_torch)
                sem_preds = points_to_voxels_torch(preds, last_pc_torch,
                    min_bound_torch, grid_dims_torch, voxel_sizes_torch)
                # Evaluate on last frame in scan (most recent one)
                sem_preds = sem_preds / torch.sum(sem_preds, dim=-1, keepdim=True)

                #For visualizaing
                test_sem_preds = torch.argmax(sem_preds, dim=-1)
                publish_pc(last_pc_torch, test_sem_preds, pred_pub,
                    model.min_bound.reshape(-1), 
                    model.max_bound.reshape(-1), 
                    model.grid_size.reshape(-1), use_mask=False)
                pdb.set_trace()
                publish_pc(last_pc_torch, sem_labels.reshape(-1), map_pub, 
                    model.min_bound.reshape(-1), 
                    model.max_bound.reshape(-1), 
                    model.grid_size.reshape(-1), use_mask=False)
                pdb.set_trace()

                batch_labels  = torch.vstack( (batch_labels, sem_labels))
                batch_preds = torch.vstack( (batch_preds, sem_preds))
                
            else:
                # Sample from free space
                fs_pc = ray_trace_batch(pc_np, labels_np, 0.3, device)
                labeled_pc = torch.from_numpy(np.vstack( (labeled_pc, fs_pc) ) ).to(device=device)
                preds = model(current_map, labeled_pc)

                # Generate masks for predictions and gt voxels
                sem_voxels = torch.from_numpy(gt_labels[b]).to(device=device)
                invalid_voxels_np = np.array(invalid_voxels[b]).astype(np.bool)

                prior_mask  = torch.logical_not(torch.all(preds==model.prior, dim=-1))
                void_mask   = sem_voxels!=0
                valid_mask  = torch.logical_not(
                    torch.from_numpy(
                        invalid_voxels_np
                    ).to(device, dtype=torch.bool)
                )
                occ_mask    = sem_voxels > 0
                full_mask = prior_mask & void_mask & valid_mask & occ_mask
                
                temp_preds = preds / torch.sum(preds, dim=-1, keepdim=True)
                publish_voxels(preds, pred_pub, model.centroids,
                    model.min_bound.reshape(-1),
                    model.max_bound.reshape(-1),
                    model.grid_size.reshape(-1) )
                    #,valid_voxels_mask=full_mask)
                pdb.set_trace()
                publish_voxels(sem_voxels, map_pub, model.centroids,
                    model.min_bound.reshape(-1),
                    model.max_bound.reshape(-1),
                    model.grid_size.reshape(-1) )
                    # ,valid_voxels_mask=full_mask)
                pdb.set_trace()

                # Add frame to running voxels
                sem_voxels  = sem_voxels[full_mask]
                preds       = preds[full_mask]
                
                sem_voxels  = sem_voxels.reshape(-1, 1)
                sem_preds   = preds / torch.sum(preds, dim=-1, keepdim=True)
                batch_labels = torch.vstack((batch_labels, sem_voxels))
                batch_preds = torch.vstack((batch_preds, sem_preds))

        batch_labels = batch_labels.reshape(-1)
        loss = criterion(batch_preds, batch_labels.long())
        loss.backward()
        optimizer.step()
   
        # Accuracy
        with torch.no_grad():
            # Softmax on expectation
            batch_preds = batch_preds / torch.sum(batch_preds, dim=-1, keepdim=True)
            max_batch_preds = torch.argmax(batch_preds, dim=-1)
            preds_masked = max_batch_preds.cpu().numpy()
            voxels_np = batch_labels.detach().cpu().numpy()
            accuracy = np.sum(preds_masked == voxels_np) / voxels_np.shape[0]

            inter, union = iou_one_frame(max_batch_preds, batch_labels, n_classes=NUM_CLASSES)
            union += 1e-6

        # Record
        writer.add_scalar(MODEL_NAME + '/Loss/Train', loss.item(), train_count)
        writer.add_scalar(MODEL_NAME + '/Accuracy/Train', accuracy, train_count)
        writer.add_scalar(MODEL_NAME + '/mIoU/Train', np.mean(inter/union), train_count)
        # print("Memory allocated ", torch.cuda.memory_allocated(device=device)/1e9)
        # print("Memory reserved ", torch.cuda.memory_reserved(device=device)/1e9)
            
        train_count += len(points)
    continue
    # writer.add_scalar(MODEL_NAME + '/Loss/Train', loss.item(), epoch)
    # writer.add_scalar(MODEL_NAME + '/Accuracy/Train', accuracy, epoch)
    # writer.add_scalar(MODEL_NAME + '/mIoU/Train', np.mean(inter/union), epoch)

    # Save model, decrease learning rate
    # my_lr_scheduler.step()
    torch.save(model.state_dict(), os.path.join(save_dir, "Epoch" + str(epoch) + ".pt"))
    print("Testing inference on validation...")

    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        counter = 0
        val_iter = 0
        num_correct = 0
        num_total = 0
        all_intersections = np.zeros(NUM_CLASSES)
        all_unions = np.zeros(NUM_CLASSES) + 1e-6 # SMOOTHING

        for points, points_labels, voxels, invalid_voxels, _ in dataloader_val:
            batch_voxels_labels = torch.zeros((0, 1), device=device, dtype=LABEL_TYPE)
            batch_preds = torch.zeros((0, NUM_CLASSES), device=device, dtype=FLOAT_TYPE)
            for f in range(len(points)):
                pc_np = np.vstack(np.array(points[f]))
                labels_np = np.vstack(np.array(points_labels[f]))
                labeled_pc = np.hstack((pc_np, labels_np))
                
                if labeled_pc.shape[0]==0: # Zero padded
                    print("continue")
                    continue

                if not USE_FREE_SPACE:
                    labeled_pc_torch = torch.from_numpy(labeled_pc)
                    preds = model(current_map, labeled_pc_torch)
                    voxels_torch = torch.from_numpy(voxels)

                    # Evaluate on last frame in scan (most recent one)
                    last_pc_torch = torch.from_numpy(np.array(points[-1]))
                    sem_preds = points_to_voxels_torch(preds, last_pc_torch,
                        rellis_ds.min_bound,
                        rellis_ds.grid_dims,
                        rellis_ds.voxel_sizes)
                    sem_labels= points_to_voxels_torch(voxels_torch, last_pc_torch,
                        rellis_ds.min_bound,
                        rellis_ds.grid_dims,
                        rellis_ds.voxel_sizes)
                    
                    sem_preds = sem_preds.reshape(-1, 1)
                    sem_preds = sem_preds / torch.sum(sem_preds, dim=-1, keepdim=True)
                    batch_voxel_labels  = torch.vstack( (batch_voxel_labels, sem_labels))
                    batch_preds = torch.vstack( (batch_preds, sem_preds))
                
                else:
                    # TODO: add in free space
                    pass

            batch_voxels_labels = batch_voxels_labels.reshape(-1)
            loss = criterion(batch_preds, batch_voxels_labels.long())
            running_loss += loss
            counter += batch_preds.shape[0]

            # Softmax on expectation
            max_batch_preds = torch.argmax(batch_preds, dim=-1)
            max_batch_preds_np = max_batch_preds.detach().cpu().numpy()
            voxels_np = batch_voxels_labels.detach().cpu().numpy()
            num_correct += np.sum(max_batch_preds_np == voxels_np)
            num_total += voxels_np.shape[0]

            inter, union = iou_one_frame(max_batch_preds, batch_voxels_labels, n_classes=NUM_CLASSES)

            try:
                all_intersections += inter
                all_unions += union
            except Exception as e:
                pdb.set_trace()

            # Record
            temp_intersections = all_intersections[all_unions > 0]
            temp_unions = all_unions[all_unions > 0]
            if val_iter%200:      
                print(f'Epoch Num: {epoch} ------ average val loss: {running_loss/counter}')
                print(f'Epoch Num: {epoch} ------ average val accuracy: {num_correct/num_total}')
                print(f'Epoch Num: {epoch} ------ val miou: {np.mean(temp_intersections / temp_unions)}')
                # writer.add_scalar(MODEL_NAME + '/Loss/Train', running_loss/counter, val_iter)
                # writer.add_scalar(MODEL_NAME + '/Accuracy/Train', num_correct/num_total, val_iter)
                # writer.add_scalar(MODEL_NAME + '/mIoU/Train', np.mean(all_intersections / all_unions), val_iter)
            val_iter += 1
            # writer.add_scalar(MODEL_NAME + '/Loss/Val', running_loss/counter, epoch)
            # writer.add_scalar(MODEL_NAME + '/Accuracy/Val', num_correct/num_total, epoch)
            # writer.add_scalar(MODEL_NAME + '/mIoU/Val', np.mean(all_intersections / all_unions), epoch)
 
        # Log Epoch
        all_intersections = all_intersections[all_unions > 0]
        all_unions = all_unions[all_unions > 0]
        print(f'Epoch Num: {epoch} ------ average val loss: {running_loss/counter}')
        print(f'Epoch Num: {epoch} ------ average val accuracy: {num_correct/num_total}')
        print(f'Epoch Num: {epoch} ------ val miou: {np.mean(all_intersections / all_unions)}')
        writer.add_scalar(MODEL_NAME + '/Loss/Val', running_loss/counter, epoch)
        writer.add_scalar(MODEL_NAME + '/Accuracy/Val', num_correct/num_total, epoch)
        writer.add_scalar(MODEL_NAME + '/mIoU/Val', np.mean(all_intersections / all_unions), epoch)

    print("Epoch ", epoch, " out of ", EPOCH_NUM, " complete.")

writer.close()