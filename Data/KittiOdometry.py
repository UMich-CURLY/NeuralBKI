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
# print(kitti_config['content'])
# print(remapdict)

# LABELS_REMAP = np.array(LABE)
# print(type(LABELS_REMAP))

def grid_ind(input_pc, labels, min_bound, max_bound, grid_size, voxel_sizes):
    '''
    Input:
        input_xyz: N * (x, y, z, c) float32 array, point cloud
    Output:
        grid_inds: N' * (x, y, z, c) int32 array, point cloud mapped to voxels
    '''
    input_xyz   = input_pc[:, :3]
    
    valid_input_mask = np.all((input_xyz < max_bound) & (input_xyz >= min_bound), axis=1)
    
    valid_xyz = input_xyz[valid_input_mask]
    labels = labels[valid_input_mask]

    grid_inds = np.floor((valid_xyz - min_bound) / voxel_sizes)
    maxes = (grid_size - 1).reshape(1, 3)
    clipped_inds = np.clip(grid_inds, np.zeros_like(maxes), maxes)

    return clipped_inds, labels, valid_xyz

# TODO: Load this from YAML
SPLIT_SEQUENCES = {
    "train": ["00"], 
    "val": ["07"], 
    "test": ["data_kitti_15"] 
}


class KittiOdomDataset(Dataset):
    """Kitti Dataset for Neural BKI project
    
    Access to the processed data, including evaluation labels predictions velodyne poses times
    """

    def __init__(self,
                grid_params,
                directory="/home/jason/kitti_odometry",
                device='cuda',
                num_frames=1,
                voxelize_input=True,
                binary_counts=False,
                random_flips=False,
                use_aug=True,
                apply_transform=True,
                remap=False,
                from_continuous=False,
                to_continuous=False,
                num_classes = 20,
                data_split='train'
                 ):
        self.use_aug = use_aug
        self.apply_transform = apply_transform

        self._grid_size = grid_params['grid_size']
        self.grid_dims = np.asarray(self._grid_size)
        self._eval_size = list(np.uint32(self._grid_size))
        self.coor_ranges = grid_params['min_bound'] + grid_params['max_bound']
        self.voxel_sizes = [abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0], 
                      abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
                      abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2]]
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])
        self.voxel_sizes = np.asarray(self.voxel_sizes)

        self.voxelize_input = voxelize_input
        self.binary_counts = binary_counts
        self._directory = directory
        self._num_frames = num_frames
        self.device = device
        self.random_flips = random_flips
        self.remap = remap
        self.from_continuous = from_continuous
        self.to_continuous = to_continuous
        self.num_classes = num_classes
        self.split = data_split

        self._velodyne_list = []
        self._label_list = []
        self._pred_list = []

        self._frames_list = []
        self._poses = np.empty((0,12))

        self._num_frames_scene = []
        self._frames_list_label = []
        self._seqs = SPLIT_SEQUENCES[self.split]

        for seq in self._seqs:
            velodyne_dir = os.path.join(self._directory, seq, 'training/pointcloud')
            label_dir = os.path.join(self._directory, seq, 'training/labels')
            if self.from_continuous:
                preds_dir = os.path.join(self._directory, seq, 'training/predictions_continuous')
            else: 
                preds_dir = os.path.join(self._directory, seq, 'training/predictions')

            self._num_frames_scene.append(len(os.listdir(label_dir)))
            frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(velodyne_dir))]
            
            frames_list_label = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(label_dir))]
            self._frames_list_label = frames_list_label

            pose = np.loadtxt(os.path.join(self._directory, seq, 'training/CameraTrajectory.txt'))[:(len(frames_list))]
            self._poses = np.vstack((self._poses, pose))

            self._frames_list.extend(frames_list)
            self._velodyne_list.extend([os.path.join(velodyne_dir, str(frame).zfill(6)+'.bin') for frame in frames_list])
            self._label_list.extend([os.path.join(label_dir, str(frame).zfill(6)+'.label') for frame in frames_list_label])
            if self.from_continuous:
                self._pred_list.extend([os.path.join(preds_dir, str(frame).zfill(6)+'.bin') for frame in frames_list]) 
            else:
                self._pred_list.extend([os.path.join(preds_dir, str(frame).zfill(6)+'.label') for frame in frames_list]) 


        self._poses = self._poses.reshape(len(frames_list), 12)

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

        R = np.array(([0, -1, 0], [0, 0, -1], [1, 0, 0])) #kitti odometry to rviz coords
        R_4 = np.zeros((4,4))
        R_4[:3,:3] = R
        R_4[3,3] = 1

        pose = np.matmul(np.linalg.inv(R_4), np.matmul(pose, R_4))
        global_pose = pose.astype(np.float32)

        return global_pose

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return sum(self._num_frames_scene)


    def __getitem__(self, idx):
        # -1 indicates no data
        # the final index is the output
        idx_2 = int(self._frames_list_label[idx])

        idx_range = self.find_horizon(idx_2)
        current_points = []
        current_labels = []

        ego_pose = self.get_pose(idx_range[-1])
        to_ego   = np.linalg.inv(ego_pose)

        aug_index = np.random.randint(0,3) # Set end idx to 6 to do rotations

        aug_mat = self.get_aug_matrix(aug_index)
        gt_labels = None

        temp_gt_labels = np.fromfile(self._label_list[idx], dtype=np.uint8).reshape((-1))

        for i in idx_range:
            if i == -1: # Zero pad
                points = np.zeros((1, 3), dtype=np.float32)
                if self.to_continuous:
                    labels = np.zeros((1,self.num_classes), dtype=np.float32)
                else:
                    labels = np.zeros((1, 1), dtype=np.uint8)
            else:
                points = np.fromfile(self._velodyne_list[i],dtype=np.float32).reshape(-1,3)
                

                if self.apply_transform:
                    global_pose   = self.get_pose(i)
                    relative_pose = np.matmul(to_ego, global_pose)
                    points = np.dot(relative_pose[:3, :3], points.T).T + relative_pose[:3, 3]

                if self.from_continuous:
                    # labels = np.fromfile(self._pred_list[i], dtype=np.float32).reshape((-1, self.num_classes))
                    labels = np.fromfile(self._pred_list[i], dtype=np.uint8).reshape((-1, self.num_classes))
                    labels = (labels / 255).astype(np.float32)
                    if not self.to_continuous:
                        labels = np.argmax(labels, axis=1).reshape((-1, 1)).astype(np.uint8)
                else:
                    labels = np.fromfile(self._pred_list[i], dtype=np.uint8).reshape((-1, 1))

                # Perform data augmentation on points
                if self.use_aug:
                    points = (aug_mat @ points.T).T

                # Filter points outside of voxel grid

                if i == idx_range[-1]:
                    grid_point_mask = np.all(
                        (points < self.max_bound) & (points >= self.min_bound), axis=1)
                    
                    points = points[grid_point_mask, :]
                    temp_gt_labels = temp_gt_labels[grid_point_mask]
                    labels = labels[grid_point_mask,:]

                    # Remove zero labels
                    void_mask = temp_gt_labels != 0
                    points = points[void_mask, :]
                    temp_gt_labels = temp_gt_labels[void_mask]
                    labels = labels[void_mask,:]

                    # Remove ignored classes (sky, person, bicycle)
                    ignored_classes = np.all((temp_gt_labels != 8, temp_gt_labels != 9, temp_gt_labels != 11), axis=0)
                    points = points[ignored_classes, :]
                    temp_gt_labels = temp_gt_labels[ignored_classes]
                    labels = labels[ignored_classes,:]

                    gt_labels = temp_gt_labels
                else:
                    grid_point_mask = np.all(
                        (points < self.max_bound) & (points >= self.min_bound), axis=1)
                    points = points[grid_point_mask, :]
                    labels = labels[grid_point_mask, :]
                    
                    # Remove zero labels
                    if self.from_continuous:
                        labels_t = np.argmax(labels, axis=1).reshape((-1, 1)).astype(np.uint8)
                    else:
                        labels_t = labels
                    void_mask = labels_t != 0
                    void_mask = void_mask.squeeze()
                    points = points[void_mask, :]
                    labels = labels[void_mask, :]

                    # Remove ignored classes (sky, person, bicycle)
                    ignored_classes = np.all((labels_t != 8,labels_t != 9,labels_t != 11), axis=0).squeeze()
                    
                    points = points[ignored_classes, :]
                    labels = labels[ignored_classes, :]

                points = points.astype(np.float32) 

            current_points.append(points)
            current_labels.append(labels)

        return current_points, current_labels, gt_labels.astype(np.uint8).reshape(-1, 1)
    
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

    def get_test_item(self, idx, get_gt=False):
            frame_id = idx # Frame ID in current scene ID
            global_pose = self.get_pose(frame_id)
            if frame_id > 0:
                prior_pose = self.get_pose(frame_id - 1)
            else: 
                prior_pose = global_pose
            
            points = np.fromfile(self._velodyne_list[frame_id], dtype=np.float32).reshape(-1, 4)[:, :3]
            if get_gt:
                gt_labels = np.fromfile(self._label_list[frame_id], dtype=np.uint8).reshape((-1))
            if self.from_continuous:
                # pred_labels = np.fromfile(self._pred_list[frame_id], dtype=np.float32).reshape((-1, self.num_classes))
                pred_labels = np.fromfile(self._pred_list[frame_id], dtype=np.uint8).reshape((-1, self.num_classes))
                pred_labels = (pred_labels / 255).astype(np.float32)
                if not self.to_continuous:
                    pred_labels = np.argmax(pred_labels, axis=1).reshape((-1, 1))
            else:
                pred_labels = np.fromfile(self._pred_list[frame_id], dtype=np.uint8).reshape((-1, 1))

            # Filter points outside of voxel grid
            grid_point_mask = np.all( (points < self.max_bound) & (points >= self.min_bound), axis=1)
            points = points[grid_point_mask, :]
            if get_gt:
                gt_labels = gt_labels[grid_point_mask]
            pred_labels = pred_labels[grid_point_mask, :]

            # Remove zero labels
            if get_gt:
                void_mask = gt_labels != 0
                points = points[void_mask, :]
                gt_labels = gt_labels[void_mask]
                pred_labels = pred_labels[void_mask]


            scene_id = 0
            if get_gt:
                return global_pose, points, pred_labels, gt_labels.astype(np.uint8).reshape(-1, 1), scene_id, frame_id
            else:
                return global_pose, points, pred_labels, None, scene_id, frame_id
            #return global_pose, points, pred_labels.astype(np.uint8).reshape(-1, 1), gt_labels.astype(np.uint8).reshape(-1, 1), scene_id, frame_id
