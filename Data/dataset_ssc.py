## Maintainer: Arthur Zhang #####
## Contact: arthurzh@umich.edu #####

import os
import pdb
import math
import numpy as np
import random
import json
from sklearn.metrics import homogeneity_completeness_v_measure

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from scipy.spatial.transform import Rotation as R
from Data.utils import *

def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def ray_trace_batch(points, labels, sample_spacing, device="cpu"):
    # curr_time = time.time()
    use_np = isinstance(points, np.ndarray)
    # Compute samples using array broadcasting
    if use_np:
        points = torch.from_numpy(points).to(device=device, dtype=torch.float16)
        labels = torch.from_numpy(labels).to(device=device, dtype=torch.uint8).reshape(-1)
    unit_vec = torch.linalg.norm(points, axis=1).reshape(-1, 1)
    unit_vec = (points / unit_vec).reshape(-1, 1, 3)

    difs = torch.arange(0.0, 100.0, sample_spacing, device=device, dtype=torch.float16).reshape(1, -1, 1)
    difs = unit_vec * difs
    new_samples = points.reshape(-1, 1, 3) - difs

    # Create labels
    new_labels = torch.ones((new_samples.shape[0], new_samples.shape[1]),
        device=device, dtype=torch.uint8) * LABELS_REMAP[-1]

    new_labels[:, 0] = labels
    new_labels = new_labels.reshape(-1)

    # Remove points with dist < 0
    vec_dists = new_samples / unit_vec

    good_samples = new_samples[vec_dists[:, :, 0] > 0].reshape(-1, 3)
    good_labels = new_labels[vec_dists[:, :, 0].reshape(-1) > 0]

    good_samples = good_samples.detach().cpu().numpy()
    good_labels = good_labels.detach().cpu().numpy().reshape(-1, 1)
    good_pc = np.hstack((good_samples, good_labels))
    # print("Elapsed time for 4 ", time.time() - curr_time)
    torch.cuda.empty_cache()
    return good_pc

class Rellis3dDataset(Dataset):
    """Rellis3D Dataset for Neural BKI project
    
    Access to the processed data, including evaluation labels predictions velodyne poses times
    """
    def __init__(self, 
        directory,
        device='cuda',
        num_frames=20,
        remap=True,
        use_aug=True,
        apply_transform=True,
        model_name="salsa",
        model_setting="train"
        ):
        '''Constructor.
        Parameters:
            directory: directory to the dataset
        '''
        self._directory = directory
        self._num_frames = num_frames
        self.device = device
        self.remap = remap
        self.use_aug = use_aug
        self.apply_transform = apply_transform

        self._scenes = [ s for s in sorted(os.listdir(self._directory)) if s.isdigit() ]
 
        self._num_scenes = len(self._scenes)
        self._num_frames_scene = 0
        self._num_labels = LABELS_REMAP.shape[0]

        param_file = os.path.join(self._directory, self._scenes[4], 'params.json')
        with open(param_file) as f:
            self._eval_param = json.load(f)
        
        self._out_dim = self._eval_param['num_channels']
        self._grid_size = self._eval_param['grid_size']
        self.grid_dims = np.asarray(self._grid_size)
        self._eval_size = list(np.uint32(self._grid_size))
        
        self.coor_ranges = self._eval_param['min_bound'] + self._eval_param['max_bound']
        self.voxel_sizes = [abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0], 
                      abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
                      abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2]]
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])
        self.voxel_sizes = np.asarray(self.voxel_sizes)

        self._velodyne_list = []
        self._label_list = []
        self._pred_list = []
        self._voxel_label_list = []
        self._occupied_list = []
        self._invalid_list = []
        self._frames_list = []
        self._timestamps = []
        self._poses = []
        self._num_frames_by_scene = []

        split_dir = os.path.join(self._directory, "pt_"+model_setting+".lst")

        # Generate list of scenes and indices to iterate over
        self._scenes_list = []
        self._index_list = []

        with open(split_dir, 'r') as split_file:
            for line in split_file:
                image_path = line.split(' ')
                image_path_lst = image_path[0].split('/')
                scene_num = image_path_lst[0]
                frame_index = int(image_path_lst[2][0:6])
                self._scenes_list.append(scene_num)
                self._index_list.append(frame_index)
        
        for scene_id in range(self._num_scenes):
            scene_name = self._scenes[scene_id]

            velodyne_dir = os.path.join(self._directory, scene_name, 'os1_cloud_node_kitti_bin')
            label_dir = os.path.join(self._directory, scene_name, 'os1_cloud_node_semantickitti_label_id')
            voxel_dir = os.path.join(self._directory, scene_name, 'voxels')
            pred_dir = os.path.join(self._directory, scene_name, model_name, 'os1_cloud_node_semantickitti_label_id')
            
            # Load all poses and frame indices regardless of mode
            self._poses.append( np.loadtxt(os.path.join(self._directory, scene_name, 'poses.txt')).reshape(-1, 12) )
            self._frames_list.append([ \
                os.path.splitext(filename)[0] for filename in sorted(os.listdir(velodyne_dir))])
            self._num_frames_by_scene.append(len(self._frames_list[scene_id]))

            # PC inputs
            self._velodyne_list.append( [os.path.join(velodyne_dir, 
                str(frame).zfill(6)+'.bin') for frame in self._frames_list[scene_id]] )
            self._label_list.append( [os.path.join(label_dir, 
                str(frame).zfill(6)+'.label') for frame in self._frames_list[scene_id]] )
            self._pred_list.append( [os.path.join(pred_dir, 
                str(frame).zfill(6)+'.label') for frame in self._frames_list[scene_id]] )
            # Voxel ground truths
            self._voxel_label_list.append( [os.path.join(voxel_dir, 
                str(frame).zfill(6)+'.label') for frame in self._frames_list[scene_id]] )
            self._occupied_list.append( [os.path.join(voxel_dir, 
                str(frame).zfill(6)+'.bin') for frame in self._frames_list[scene_id]] )
            self._invalid_list.append( [os.path.join(voxel_dir, 
                str(frame).zfill(6)+'.invalid') for frame in self._frames_list[scene_id]] )

        # Get number of frames to iterate over
        self._num_frames_scene = len(self._index_list)

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return self._num_frames_scene
    
    def collate_fn(self, data):
        output_batch = [bi[0] for bi in data]
        label_batch = [bi[1] for bi in data]
        voxel_batch = [bi[2] for bi in data]
        invalid_batch = [bi[3] for bi in data]
        occupied_batch = [bi[4] for bi in data]
        return output_batch, label_batch, voxel_batch, invalid_batch, occupied_batch

    def points_to_voxels(self, voxel_grid, points):
        # Valid voxels (make sure to clip)
        valid_point_mask= np.all(
            (points < self.max_bound) & (points >= self.min_bound), axis=1)
        valid_points = points[valid_point_mask, :]

        voxels = np.floor((valid_points - self.min_bound) / self.voxel_sizes).astype(np.int)
        # Clamp to account for any floating point errors
        maxes = np.reshape(self.grid_dims - 1, (1, 3))
        mins = np.zeros_like(maxes)
        voxels = np.clip(voxels, mins, maxes).astype(np.int)

        voxel_grid = voxel_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
        return voxel_grid

    def get_file_path(self, idx):
        print(self._frames_list[idx])

    def get_voxel_aug(self, t, state):
        if state == 1:
            aug_t = np.flip(t, [0]) # XZ
        elif state == 2:
            aug_t = np.flip(t, [1]) # YZ
        elif state == 3:
            aug_t = np.rot90(t, 1, [0, 1])
        elif state == 4:
            aug_t = np.rot90(t, 2, [0, 1])
        elif state == 5:
            aug_t = np.rot90(t, 3, [0, 1])
        else:
            aug_t = t

        return aug_t

    def get_aug_matrix(self, trans):
        """
            trans - 1 or 2 specifies reflection about XZ or YZ plane
                    any other value gives rotation matrix
                    Double checked with rotation matrix calculator
        """
        if trans==1:
            trans = np.eye(3)
            trans[1][1] = -1
        elif trans==2:
            trans = np.eye(3)
            trans[0][0] = -1
        else:
            if trans==0:
                angle = 0
            else:
                angle = (trans-2)*90
            trans = R.from_euler('z', angle, degrees=True).as_matrix()

        return trans

    def get_pose(self, scene_id, frame_id):
        pose = np.zeros((4, 4))
        pose[3, 3] = 1
        pose[:3, :4] = self._poses[scene_id][frame_id].reshape(3, 4)
        return pose
    
    def __getitem__(self, idx):
        scene_name  = self._scenes_list[idx]
        scene_id    = int(scene_name)       # Scene ID
        frame_id    = self._index_list[idx] # Frame ID in current scene ID
        
        idx_range = self.find_horizon(scene_id, frame_id)

        current_points = []
        current_labels = []

        voxels = np.fromfile(
                self._voxel_label_list[scene_id][idx_range[-1]], dtype=np.uint8
            ).reshape(self.grid_dims.astype(np.int))
        if self.remap:
            voxels = LABELS_REMAP[voxels].astype(np.uint8)

        invalid_voxels = unpack(np.fromfile(self._invalid_list[scene_id][idx_range[-1]], 
                        dtype=np.uint8)).reshape(self.grid_dims.astype(np.int))
        occupied_voxels= unpack(np.fromfile(
                self._occupied_list[scene_id][idx_range[-1]], dtype=np.uint8
            )).reshape(self.grid_dims.astype(np.int))

        ego_pose = self.get_pose(scene_id, idx_range[-1])
        to_ego   = np.linalg.inv(ego_pose)
        
        aug_index = np.random.randint(0,6) # Set end idx to 6 to do rotations
        aug_mat = self.get_aug_matrix(aug_index)

        for i in idx_range:
            if i == -1: # Zero pad
                points = np.zeros((1, 3), dtype=np.float16)
                labels = np.zeros((1,), dtype=np.uint8)
            else:
                points = np.fromfile(self._velodyne_list[scene_id][i], 
                    dtype=np.float32).reshape(-1,4)[:, :3]
                gt_labels = np.fromfile(self._label_list[scene_id][i], 
                    dtype=np.uint32).reshape((-1)).astype(np.uint8)
                labels = np.fromfile(self._pred_list[scene_id][i], 
                    dtype=np.uint32).reshape((-1)).astype(np.uint8)
                
                if self.apply_transform:
                    to_world   = self.get_pose(scene_id, i)
                    relative_pose = np.matmul(to_ego, to_world)
                    points = np.dot(relative_pose[:3, :3], points.T).T + relative_pose[:3, 3]

                # Filter points outside of voxel grid
                grid_point_mask= np.all(
                    (points < self.max_bound) & (points >= self.min_bound), axis=1)
                points = points[grid_point_mask, :]
                gt_labels = gt_labels[grid_point_mask]
                labels = labels[grid_point_mask]
                
                if self.remap:
                    gt_labels = LABELS_REMAP[gt_labels].astype(np.uint8)
                    labels = LABELS_REMAP[labels].astype(np.uint8)

                # Ego vehicle = 0
                valid_point_mask = np.isin(gt_labels, DYNAMIC_LABELS, invert=True)
                points = points[valid_point_mask]
                labels = labels[valid_point_mask]

                # Perform data augmentation on points
                if self.use_aug:  
                    points = (aug_mat @ points.T).T

                labels = labels.reshape(-1, 1)

                points = points.astype(np.float16)
                labels = labels.astype(np.uint8)

            current_points.append(points)
            current_labels.append(labels)

        # Align voxels with augmented points
        if self.use_aug:
            voxels = self.get_voxel_aug(voxels, aug_index)
            invalid_voxels = self.get_voxel_aug(invalid_voxels, aug_index)
            occupied_voxels= self.get_voxel_aug(occupied_voxels, aug_index)

        return current_points, current_labels, voxels, invalid_voxels, occupied_voxels

    def find_horizon(self, scene_id, idx):
        end_idx = idx
        idx_range = np.arange(idx- self._num_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[scene_id][end_idx]) \
            - int(self._frames_list[scene_id][i]) for i in idx_range])
        good_diffs = -1 * (np.arange(- self._num_frames, 0) + 1)
        idx_range[good_diffs != diffs] = -1

        return idx_range

