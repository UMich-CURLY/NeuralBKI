## Maintainer: Arthur Zhang #####
## Contact: arthurzh@umich.edu #####

import os
import pdb
import math
import numpy as np
import random
import json

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

class Rellis3dDataset(Dataset):
    """Rellis3D Dataset for Neural BKI project
    
    Access to the processed data, including evaluation labels predictions velodyne poses times
    """
    def __init__(self, 
        directory,
        device='cuda',
        num_frames=20,
        voxelize_input=False,
        remap=True,
        use_gt=True,
        use_aug=True,
        apply_transform=True,
        model_name="salsa"
        ):
        '''Constructor.
        Parameters:
            directory: directory to the dataset
        '''

        self.voxelize_input = voxelize_input
        self._directory = directory
        self._num_frames = num_frames
        self.device = device
        self.remap = remap
        self.use_gt = use_gt
        self.use_aug = use_aug
        self.apply_transform = apply_transform
        
        self._scenes = [ s for s in sorted(os.listdir(self._directory)) if s.isdigit() ]
 
        self._num_scenes = len(self._scenes)
        self._num_frames_scene = []

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
        self._voxel_list = []
        self._invalid_list = []
        self._frames_list = []
        self._timestamps = []
        self._poses = [] 

        # for scene in self._scenes:
        for i in range(0, 1):
            scene = self._scenes[-1]
            velodyne_dir = os.path.join(self._directory, scene, 'os1_cloud_node_kitti_bin')
            label_dir = os.path.join(self._directory, scene, 'os1_cloud_node_semantickitti_label_id')
            voxel_dir = os.path.join(self._directory, scene, 'voxels')
            pred_dir = os.path.join(self._directory, scene, model_name, 'os1_cloud_node_semantickitti_label_id')
            
            self._num_frames_scene.append(len(os.listdir(velodyne_dir)))

            frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(velodyne_dir))]

            self._frames_list.extend(frames_list)
            self._velodyne_list.extend([os.path.join(velodyne_dir, str(frame).zfill(6)+'.bin') for frame in frames_list])
            self._label_list.extend([os.path.join(label_dir, str(frame).zfill(6)+'.label') for frame in frames_list])
            self._voxel_list.extend([os.path.join(voxel_dir, str(frame).zfill(6)+'.label') for frame in frames_list])
            self._pred_list.extend([os.path.join(pred_dir, str(frame).zfill(6)+'.label') \
                for frame in frames_list])
            self._invalid_list.extend(
                [os.path.join(voxel_dir, str(frame).zfill(6)+'.invalid') \
                for frame in frames_list]
            )
            self._poses.append(np.loadtxt(os.path.join(self._directory, scene, 'poses.txt')))

        self._poses = np.array(self._poses).reshape(-1, 12)
        
        self._cum_num_frames = np.cumsum(np.array(self._num_frames_scene) - self._num_frames + 1)

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return sum(self._num_frames_scene)
    
    def collate_fn(self, data):
        output_batch = [bi[0] for bi in data]
        label_batch = [bi[1] for bi in data]
        voxel_batch = [bi[2] for bi in data]
        invalid_batch = [bi[3] for bi in data]
        return output_batch, label_batch, voxel_batch, invalid_batch
    
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
    
    def __getitem__(self, idx):
        # -1 indicates no data
        # print("idx ", idx)
        # the final index is the output
        idx_range = self.find_horizon(idx)
         
        current_points = []
        current_labels = []

        voxels = np.fromfile(
            self._voxel_list[idx_range[-1]], dtype=np.uint16
        ).reshape(self.grid_dims.astype(np.int))
        voxels = LABELS_REMAP[voxels].astype(np.uint32)
        invalid_voxels = np.zeros_like(voxels, dtype=np.uint8)

        curr_pose_mat = self._poses[idx_range[-1]].reshape(3, 4)
        curr_pose_rot   = curr_pose_mat[0:3, 0:3].T # Global to current rot R^T
        curr_pose_trans = -curr_pose_rot @ curr_pose_mat[:, 3] # Global to current trans (-R^T * t)
        
        aug_index = np.random.randint(0,6)

        aug_mat = self.get_aug_matrix(aug_index)
        for i in idx_range:
            if i == -1: # Zero pad
                points = np.zeros((1, 3), dtype=np.float32)
                labels = np.zeros((1,), dtype=np.uint32)
            else:
                points = np.fromfile(self._velodyne_list[i],dtype=np.float32).reshape(-1,4)[:, :3]

                if self.apply_transform:
                    prev_pose_mat = self._poses[i].reshape(3, 4)
                    prev_pose_rot = prev_pose_mat[0:3, 0:3]
                    prev_pose_trans= prev_pose_mat[:, 3]
                    # pdb.set_trace()
                    points_in_global = ((prev_pose_rot @ points.T).T + prev_pose_trans)
                    points = (curr_pose_rot @ points_in_global.T).T + curr_pose_trans
                if not self.use_gt:
                    preds = np.fromfile(self._pred_list[i], dtype=np.uint32).reshape((-1))
                else:
                    preds = np.fromfile(self._label_list[i], dtype=np.uint32).reshape((-1))
                labels = preds & 0xFFFF

                gt = np.fromfile(self._pred_list[i], dtype=np.uint32).reshape((-1))
                # print("Accuracy ", np.sum(gt==preds) / gt.shape[0])
            
                current_invalid_voxels = unpack(
                    np.fromfile(self._invalid_list[i], dtype=np.uint8)
                ).reshape(self.grid_dims.astype(np.int))
                invalid_voxels = invalid_voxels | current_invalid_voxels

            if self.remap:
                labels = LABELS_REMAP[labels].astype(np.uint32)

            # Ego vehicle = 0
            non_void = labels != 0
            points = points[non_void]
            labels = labels[non_void]

            # Limit points to grid size
            valid_point_mask = np.all(
                (points < self.max_bound) & (points >= self.min_bound), axis=1
            )
            points = points[valid_point_mask, :]
            labels = labels[valid_point_mask]

            # Perform data augmentation
            if self.use_aug:  
                points = (aug_mat @ points.T).T

            # Save augmentation to avoid duplicating voxel data
            current_points.append(points)
            current_labels.append(labels)

        # Align voxels with augmented pointss
        if self.use_aug:
            voxels = self.get_voxel_aug(voxels, aug_index)
            invalid_voxels = self.get_voxel_aug(invalid_voxels, aug_index)

        return current_points, current_labels, voxels, invalid_voxels
    
    def find_horizon(self, idx):
        end_idx = idx

        idx_range = np.arange(idx-self._num_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[end_idx]) - int(self._frames_list[i]) for i in idx_range])
        good_diffs = -1 * (np.arange(-self._num_frames, 0) + 1)
        # print("idx range ", idx_range)
        idx_range[good_diffs != diffs] = -1

        return idx_range