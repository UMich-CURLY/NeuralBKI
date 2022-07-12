## Maintainer: Arthur Zhang #####
## Contact: arthurzh@umich.edu #####

import os
import pdb
import math
import numpy as np
import random
import json
import yaml
from sklearn.metrics import homogeneity_completeness_v_measure

import torch
from torch import gt
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
        grid_params,
        directory="/frog-drive/jingyuso/dataset/kitti",
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

        data_params_file = os.path.join(os.getcwd(), "Config", "rellis.yaml")
        with open(data_params_file, "r") as stream:
            try:
                data_params = yaml.safe_load(stream)
                self._num_labels = data_params["num_classes"]
                max_label = max([i for i in data_params["LABELS_REMAP"].keys()])
                self.LABELS_REMAP = np.zeros(max_label + 1, dtype=np.long)
                for v,k in data_params["LABELS_REMAP"].items():
                    self.LABELS_REMAP[v] = k
            except yaml.YAMLError as exc:
                print(exc)

        self._grid_size = grid_params['grid_size']
        self.grid_dims = np.asarray(self._grid_size)

        self.coor_ranges = grid_params['min_bound'] + grid_params['max_bound']
        self.voxel_sizes = np.asarray([abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0],
                            abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
                            abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2]])
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])

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
            pred_dir = os.path.join(self._directory, scene_name, model_name, 'os1_cloud_node_semantickitti_label_id')
            
            # Load all poses and frame indices regardless of mode
            self._poses.append(np.loadtxt(os.path.join(self._directory, scene_name, 'poses.txt')).reshape(-1, 12) )
            self._frames_list.append([os.path.splitext(filename)[0] for filename in sorted(os.listdir(velodyne_dir))])
            self._num_frames_by_scene.append(len(self._frames_list[scene_id]))

            # PC inputs
            self._velodyne_list.append( [os.path.join(velodyne_dir, 
                str(frame).zfill(6)+'.bin') for frame in self._frames_list[scene_id]] )
            self._label_list.append( [os.path.join(label_dir, 
                str(frame).zfill(6)+'.label') for frame in self._frames_list[scene_id]] )
            self._pred_list.append( [os.path.join(pred_dir, 
                str(frame).zfill(6)+'.label') for frame in self._frames_list[scene_id]] )

        # Get number of frames to iterate over
        self._num_frames_scene = len(self._index_list)

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return self._num_frames_scene
    
    def collate_fn(self, data):
        points_batch = [bi[0] for bi in data]
        label_batch = [bi[1] for bi in data]
        gt_label_batch = [bi[2] for bi in data]
        return points_batch, label_batch, gt_label_batch

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

        ego_pose = self.get_pose(scene_id, idx_range[-1])
        to_ego   = np.linalg.inv(ego_pose)
        
        aug_index = np.random.randint(0,3) # Set end idx to 6 to do rotations

        aug_mat = self.get_aug_matrix(aug_index)

        gt_labels = None
        for i in idx_range:
            if i == -1: # Zero pad
                points = np.zeros((1, 3), dtype=np.float16)
                labels = np.zeros((1,), dtype=np.uint8)
            else:
                points = np.fromfile(self._velodyne_list[scene_id][i], dtype=np.float32).reshape(-1,4)[:, :3]
                
                if self.apply_transform:
                    to_world   = self.get_pose(scene_id, i)
                    to_world = to_world
                    relative_pose = np.matmul(to_ego, to_world)

                    points = np.dot(relative_pose[:3, :3], points.T).T + relative_pose[:3, 3]

                temp_gt_labels = np.fromfile(self._label_list[scene_id][i], dtype=np.uint32).reshape((-1)).astype(np.uint8)
                labels = np.fromfile(self._pred_list[scene_id][i], dtype=np.uint32).reshape((-1)).astype(np.uint8)

                # Perform data augmentation on points
                if self.use_aug:
                    points = (aug_mat @ points.T).T

                # Filter points outside of voxel grid
                grid_point_mask = np.all(
                    (points < self.max_bound) & (points >= self.min_bound), axis=1)
                points = points[grid_point_mask, :]
                temp_gt_labels = temp_gt_labels[grid_point_mask]
                labels = labels[grid_point_mask]

                # Remove zero labels
                void_mask = temp_gt_labels != 0
                points = points[void_mask, :]
                temp_gt_labels = temp_gt_labels[void_mask]
                labels = labels[void_mask]
                
                if self.remap:
                    temp_gt_labels = self.LABELS_REMAP[temp_gt_labels].astype(np.uint8)
                    labels = self.LABELS_REMAP[labels].astype(np.uint8)

                if i == idx_range[-1]:
                    gt_labels = temp_gt_labels

                labels = labels.reshape(-1, 1)

                points = points.astype(np.float32) #[:, [1, 0, 2]]
                labels = labels.astype(np.uint8)

            current_points.append(points)
            current_labels.append(labels)

        return current_points, current_labels, gt_labels.astype(np.uint8).reshape(-1, 1)

    def find_horizon(self, scene_id, idx):
        end_idx = idx
        idx_range = np.arange(idx- self._num_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[scene_id][end_idx])
                            - int(self._frames_list[scene_id][i]) for i in idx_range])
        good_diffs = -1 * (np.arange(- self._num_frames, 0) + 1)
        idx_range[good_diffs != diffs] = -1

        return idx_range

