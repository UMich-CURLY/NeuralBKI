import os
import pdb
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np

# Torch imports
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Custom Imports
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
    
home_dir = os.path.expanduser('~')
dataset_loc = os.path.join(home_dir, "Data/Rellis-3D")

#CONSTANTS
SEED = 42
NUM_CLASSES = colors.shape[0]
TRAIN_DIR = dataset_loc
NUM_FRAMES = 8
MODEL_NAME = "DiscreteBKI"
model_name = MODEL_NAME + "_" + str(NUM_CLASSES)

MODEL_RUN_DIR = os.path.join("Models", "Runs", model_name)
TRIAL_NUM = str(len(os.listdir(MODEL_RUN_DIR)))
NUM_WORKERS = 16
EPOCH_NUM = 500
FLOAT_TYPE = torch.float32
LABEL_TYPE = torch.uint8


#Model Parameters
class_frequencies = CLASS_COUNTS_REMAPPED
class_frequencies[DYNAMIC_LABELS] = 0 # dynamic object and void are filtered out
epsilon_w = 1e-6  # eps to avoid zero division
weights = torch.from_numpy( \
    (1 / np.log(class_frequencies + epsilon_w) ) *
    np.sum(class_frequencies)/class_frequencies.shape[0]
).to(torch.float32)
criterion = FocalLoss(2, weights, device)
    # nn.CrossEntropyLoss(weight=weights.to(device))

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

rellis_ds = Rellis3dDataset(directory=TRAIN_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=True)
dataloader = DataLoader(rellis_ds, batch_size=B, shuffle=True, collate_fn=rellis_ds.collate_fn, num_workers=NUM_WORKERS)

rellis_ds_val  = Rellis3dDataset(directory=TRAIN_DIR, device=device, num_frames=NUM_FRAMES, remap=True, use_aug=True, model_setting="val")
dataloader_val = DataLoader(rellis_ds, batch_size=B, shuffle=True, collate_fn=rellis_ds.collate_fn, num_workers=NUM_WORKERS)

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
current_map = model.initialize_grid()
batch_idx = 0

for epoch in range(EPOCH_NUM):
    # Training
    model.train()

    for points, points_labels, voxels, invalid_voxels, _ in dataloader:
        batch_voxels_labels = torch.zeros((0, 1), device=device, dtype=LABEL_TYPE)
        batch_preds = torch.zeros((0, NUM_CLASSES), device=device, dtype=FLOAT_TYPE)

        optimizer.zero_grad()
        for f in range(len(points)):
            pc_np = np.vstack(np.array(points[f]))
            labels_np = np.vstack(np.array(points_labels[f]))
            labeled_pc = np.hstack((pc_np, labels_np))

             # Sample from free space
            fs_pc = ray_trace_batch(pc_np, labels_np, 0.3, device)
            labeled_pc = torch.from_numpy(np.vstack( (labeled_pc, fs_pc) ) ).to(device=device)

            if labeled_pc.shape[0]==0: # Zero padded
                print("continue")
                continue
            
            preds = model(current_map, labeled_pc)

            prior_mask = torch.logical_not(torch.all(preds==model.prior, dim=-1))

            voxels_np = np.array(voxels[f]).astype(np.uint8)
            voxels_labels= torch.from_numpy(
                voxels_np
            ).to(device)
            void_mask = voxels_labels!=0
            
            invalid_voxels_np = np.array(invalid_voxels[f]).astype(np.bool)
            valid_voxels_mask = torch.logical_not(
                torch.from_numpy(
                    invalid_voxels_np
                ).to(device, dtype=torch.bool)
            )

            # Exclude free space, invalid voxels, and nonupdated map cells
            voxels_mask = void_mask & valid_voxels_mask & prior_mask
            valid_voxels_labels = voxels_labels[voxels_mask]
            preds_masked = preds[voxels_mask]

            valid_voxels_labels       = valid_voxels_labels.view(-1, 1)
            batch_voxels_labels = torch.vstack((batch_voxels_labels, valid_voxels_labels))
            expected_preds = preds_masked / torch.sum(preds_masked, dim=-1, keepdim=True)
            batch_preds = torch.vstack((batch_preds, expected_preds))
        
        batch_voxels_labels = batch_voxels_labels.reshape(-1)
        loss = criterion(batch_preds, batch_voxels_labels.long())
        loss.backward()
        optimizer.step()

        # Accuracy
        with torch.no_grad():
            # Softmax on expectation
            max_batch_preds = torch.argmax(batch_preds, dim=-1)
            preds_masked = max_batch_preds.cpu().numpy()
            voxels_np = batch_voxels_labels.detach().cpu().numpy()
            accuracy = np.sum(preds_masked == voxels_np) / voxels_np.shape[0]

            inter, union = iou_one_frame(max_batch_preds, batch_voxels_labels, n_classes=NUM_CLASSES)
            inter = inter[union > 0]
            union = union[union > 0]

        # Record
        writer.add_scalar(MODEL_NAME + '/Loss/Train', loss.item(), train_count)
        writer.add_scalar(MODEL_NAME + '/Accuracy/Train', accuracy, train_count)
        writer.add_scalar(MODEL_NAME + '/mIoU/Train', np.mean(inter/union), train_count)
        # print("Memory allocated ", torch.cuda.memory_allocated(device=device)/1e9)
        # print("Memory reserved ", torch.cuda.memory_reserved(device=device)/1e9)
            
        train_count += len(points)

    # Save model, decrease learning rate
    my_lr_scheduler.step()
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
        all_unions = np.zeros(NUM_CLASSES) # SMOOTHING

        for points, points_labels, voxels, invalid_voxels, _ in dataloader_val:
            batch_voxels_labels = torch.zeros((0, 1), device=device, dtype=LABEL_TYPE)
            batch_preds = torch.zeros((0, NUM_CLASSES), device=device, dtype=FLOAT_TYPE)
            for f in range(len(points)):
                pc_np = np.vstack(np.array(points[f]))
                labels_np = np.vstack(np.array(points_labels[f]))
                labeled_pc = np.hstack((pc_np, labels_np))

                # Sample from free space
                fs_pc = ray_trace_batch(pc_np, labels_np, 0.3, device)
                labeled_pc = torch.from_numpy(np.vstack( (labeled_pc, fs_pc) ) ).to(device=device)
                
                if labeled_pc.shape[0]==0: # Zero padded
                    print("continue")
                    continue
                
                preds = model(current_map, labeled_pc)

                prior_mask = torch.logical_not(torch.all(preds==model.prior, dim=-1))

                voxels_np = np.array(voxels[f]).astype(np.uint8)
                voxels_labels= torch.from_numpy(
                    voxels_np
                ).to(device)
                occupied_voxels_mask = (voxels_labels!=0) #& (voxels_labels!=20)
                
                invalid_voxels_np = np.array(invalid_voxels[f]).astype(np.bool)
                valid_voxels_mask = torch.logical_not(
                    torch.from_numpy(
                        invalid_voxels_np
                    ).to(device, dtype=torch.bool)
                )

                # Exclude free space, invalid voxels, and nonupdated map cells
                voxels_mask = occupied_voxels_mask & valid_voxels_mask & prior_mask
                valid_voxels_labels = voxels_labels[voxels_mask]
                preds_masked = preds[voxels_mask]

                valid_voxels_labels       = valid_voxels_labels.view(-1, 1)
                batch_voxels_labels = torch.vstack((batch_voxels_labels, valid_voxels_labels))
                expected_preds = preds_masked / torch.sum(preds_masked, dim=-1, keepdim=True)
                batch_preds = torch.vstack((batch_preds, expected_preds))

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