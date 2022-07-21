## Maintainer: Jingyu Song #####
## Contact: jingyuso@umich.edu #####


import os
import numpy as np
# from utils import laserscan
import yaml
from torch.utils.data import Dataset
import torch
# import spconv
import math
from scipy.spatial.transform import Rotation as R

config_file = os.path.join('Config/semantic_kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))
remapdict = kitti_config["learning_map"]
# print(kitti_config['content'])
# print(remapdict)
LABELS_REMAP = kitti_config["learning_map"]
LABEL_INV_REMAP = kitti_config["learning_map_inv"]
# LABELS_REMAP = np.array(LABE)
# print(type(LABELS_REMAP))



SPLIT_SEQUENCES = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "valid": ["08"],
    "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
}


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



class KittiDataset(Dataset):
    """Kitti Dataset for Neural BKI project
    
    Access to the processed data, including evaluation labels predictions velodyne poses times
    """

    def __init__(self, 
            grid_params,
            directory="/frog-drive/jingyuso/dataset/kitti",
            device='cuda',
            num_frames=4,
            voxelize_input=True,
            binary_counts=False,
            random_flips=False,
            use_aug=True,
            apply_transform=True,
            remap=True,
            split='train'
            ):
        

        self.use_aug = use_aug
        self.apply_transform = apply_transform


        self._grid_size = grid_params['grid_size']
        self.grid_dims = np.asarray(self._grid_size)
        self._eval_size = list(np.uint32(self._grid_size))
        # self.coor_ranges = [-25.6,-25.6,-2] + [25.6,25.6,4.4]
        self.coor_ranges = grid_params['min_bound'] + grid_params['max_bound']
        self.voxel_sizes = [abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0], 
                      abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
                      abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2]]
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])
        self.voxel_sizes = np.asarray(self.voxel_sizes)

        self.voxelize_input = voxelize_input
        self.binary_counts = binary_counts
        self._directory = os.path.join(directory, 'sequences')
        self._num_frames = num_frames
        self.device = device
        self.random_flips = random_flips
        self.remap = remap
        self.split = split

        self._remap_lut = self.get_remap_lut()

        self._velodyne_list = []
        self._label_list = []
        self._pred_list = []
        self._eval_labels = []
        self._eval_valid = []
        self._frames_list = []
        self._timestamps = []
        self._poses = np.empty((0,12))

        self._num_frames_scene = []

        self._seqs = SPLIT_SEQUENCES[self.split]

        for seq in self._seqs:
            velodyne_dir = os.path.join(self._directory, seq, 'velodyne')
            label_dir = os.path.join(self._directory, seq, 'labels')
            preds_dir = os.path.join(self._directory, seq, 'preds') # jingyu add preds
            # eval_dir = os.path.join(self._directory, seq, 'voxels')
            self._num_frames_scene.append(len(os.listdir(velodyne_dir)))
            frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(velodyne_dir))]
            
            pose = np.loadtxt(os.path.join(self._directory, seq, 'poses.txt'))
            self._poses = np.vstack((self._poses, pose))

            self._frames_list.extend(frames_list)
            self._velodyne_list.extend([os.path.join(velodyne_dir, str(frame).zfill(6)+'.bin') for frame in frames_list])
            self._label_list.extend([os.path.join(label_dir, str(frame).zfill(6)+'.label') for frame in frames_list])
            self._pred_list.extend([os.path.join(preds_dir, str(frame).zfill(6)+'.label') for frame in frames_list]) # jingyu add preds
            # self._eval_labels.extend([os.path.join(eval_dir, str(frame).zfill(6)+'.label') for frame in frames_list])
            # self._eval_valid.extend([os.path.join(eval_dir, str(frame).zfill(6) + '.invalid') for frame in frames_list])
        # assert len(self._eval_labels) == np.sum(self._num_frames_scene), f"inconsitent number of frames detected, check the dataset"
        assert len(self._velodyne_list) == np.sum(self._num_frames_scene), f"inconsitent number of frames detected, check the dataset"
        # self._poses = np.concatenate(self._poses)
        
        self._poses = self._poses.reshape(sum(self._num_frames_scene), 12)
        


    def collate_fn(self, data):
        points_batch = [bi[0] for bi in data]
        label_batch = [bi[1] for bi in data]
        gt_label_batch = [bi[2] for bi in data]
        return points_batch, label_batch, gt_label_batch
    
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

    def get_pose(self, frame_id):
        pose = np.zeros((4, 4))
        pose[3, 3] = 1
        pose[:3, :4] = self._poses[frame_id,:].reshape(3, 4)
        return pose

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return sum(self._num_frames_scene)
    

    def get_inv_remap_lut(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''

        # make lookup table for mapping
        maxkey = max(LABEL_INV_REMAP.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
        remap_lut[list(LABEL_INV_REMAP.keys())] = list(LABEL_INV_REMAP.values())

        return remap_lut

    def get_remap_lut(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''

        # make lookup table for mapping
        maxkey = max(LABELS_REMAP.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(LABELS_REMAP.keys())] = list(LABELS_REMAP.values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut

    def __getitem__(self, idx):
        # -1 indicates no data
        # the final index is the output
        idx_range = self.find_horizon(idx)

        current_points = []
        current_labels = []

        ego_pose = self.get_pose(idx_range[-1])
        to_ego   = np.linalg.inv(ego_pose)

        aug_index = np.random.randint(0,3) # Set end idx to 6 to do rotations

        aug_mat = self.get_aug_matrix(aug_index)
        gt_labels = None

        
        # if self.voxelize_input:
        #     current_horizon = np.zeros((idx_range.shape[0], int(self.grid_dims[0]), int(self.grid_dims[1]), int(self.grid_dims[2])), dtype=np.float32)
        # else:
        #     current_horizon = []

        t_i = 0
        for i in idx_range:
            if i == -1: # Zero pad
                points = np.zeros((1, 3), dtype=np.float32)
                labels = np.zeros((1,), dtype=np.uint8)
            else:
                points = np.fromfile(self._velodyne_list[i],dtype=np.float32).reshape(-1,4)[:, :3]
                if self.apply_transform:
                    to_world   = self.get_pose(i)
                    to_world = to_world
                    relative_pose = np.matmul(to_ego, to_world)

                    points = np.dot(relative_pose[:3, :3], points.T).T + relative_pose[:3, 3]
                temp_gt_labels = np.fromfile(self._label_list[i], dtype=np.uint32).reshape((-1)).astype(np.uint8)
                labels = np.fromfile(self._pred_list[i], dtype=np.uint32).reshape((-1)).astype(np.uint8)

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
                    temp_gt_labels = self._remap_lut[temp_gt_labels].astype(np.uint8)
                    labels = self._remap_lut[labels].astype(np.uint8)
                if i == idx_range[-1]:
                    gt_labels = temp_gt_labels
                
                labels = labels.reshape(-1, 1)

                points = points.astype(np.float32) #[:, [1, 0, 2]]
                labels = labels.astype(np.uint8)
            
            current_points.append(points)
            current_labels.append(labels)

        return current_points, current_labels, gt_labels.astype(np.uint8).reshape(-1, 1)
            
        #     t_i += 1
        
        # output = np.fromfile(self._eval_labels[idx_range[-1]],dtype=np.uint16).reshape(self._eval_size).astype(np.uint8)
        # counts = unpack(np.fromfile(self._eval_valid[idx_range[-1]],dtype=np.uint8)).reshape(self._eval_size)
        
        # if self.voxelize_input and self.random_flips:
        #     # X flip
        #     rand_num = np.random.randint(3)
        #     if rand_num == 0:
        #         output = np.flip(output, axis=0)
        #         counts = np.flip(counts, axis=0)
        #         current_horizon = np.flip(current_horizon, axis=1) # Because there is a time dimension
        #     # Y Flip
        #     if rand_num == 1:
        #         output = np.flip(output, axis=1)
        #         counts = np.flip(counts, axis=1)
        #         current_horizon = np.flip(current_horizon, axis=2) # Because there is a time dimension
            
        #     # add X/Y flip
        #     if rand_num == 2:
        #         output = np.flip(np.flip(output, axis=0), axis=1)
        #         counts = np.flip(np.flip(counts, axis=0), axis=1)
        #         current_horizon = np.flip(np.flip(current_horizon, axis=1), axis=2) # Because there is a time dimension
                
        # if self.remap:
        #     output = self._remap_lut[output].astype(np.uint8)
        #     # print(type(output))

        # return current_horizon, output, counts
        
        # no enough frames
    
    def find_horizon(self, idx):
        end_idx = idx
        idx_range = np.arange(idx-self._num_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[end_idx]) - int(self._frames_list[i]) for i in idx_range])
        good_difs = -1 * (np.arange(-self._num_frames, 0) + 1)
        
        idx_range[good_difs != diffs] = -1

        return idx_range
    
    def points_to_voxels(self, voxel_grid, points, t_i):
        # Valid voxels (make sure to clip)
        valid_point_mask= np.all(
            (points < self.max_bound) & (points >= self.min_bound), axis=1)
        valid_points = points[valid_point_mask, :]
        voxels = np.floor((valid_points - self.min_bound) / self.voxel_sizes).astype(int)
        # Clamp to account for any floating point errors
        maxes = np.reshape(self.grid_dims - 1, (1, 3))
        mins = np.zeros_like(maxes)
        voxels = np.clip(voxels, mins, maxes).astype(int)
        # This line is needed to create a mask with number of points, not just binary occupied
        if self.binary_counts:
             voxel_grid[t_i, voxels[:, 0], voxels[:, 1], voxels[:, 2]] += 1
        else:
            unique_voxels, counts = np.unique(voxels, return_counts=True, axis=0)
            unique_voxels = unique_voxels.astype(int)
            voxel_grid[t_i, unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]] += counts
        return voxel_grid



# ds = KittiDataset('/media/sdb1/kitti/')
# len(ds)
# print(ds[0])



# current_horizon, output, counts = ds[20]
# print(np.unique(output, return_counts=True))

# print(np.unique(counts, return_counts=True))