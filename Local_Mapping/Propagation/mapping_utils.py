# This file contains classes for local and global offline mapping (not running semantic prediction)
import torch
import torch.nn.functional as F
import numpy as np
import time


def axis_limits(offset, axis_size):
    min_bound = 0
    max_bound = axis_size
    to_min = min_bound - offset
    to_max = max_bound - offset
    from_min = min_bound + offset
    from_max = max_bound + offset
    bounds = torch.clamp(torch.stack([to_min, to_max, from_min, from_max]), min=min_bound, max=max_bound).long()
    return bounds


def grid_limits(grid, voxel_translation):
    x_bounds = axis_limits(voxel_translation[0], grid.shape[0])
    y_bounds = axis_limits(voxel_translation[1], grid.shape[1])
    z_bounds = axis_limits(voxel_translation[2], grid.shape[2])
    return [x_bounds, y_bounds, z_bounds]


class TransformWorldStatic(torch.nn.Module):
    def __init__(self, voxel_sizes, prior=1e-6):
        super().__init__()
        self.initial_pose = None
        self.global_pose = None
        self.voxel_sizes = voxel_sizes
        self.prior = prior
        self.ego_to_map = None
        self.translation = torch.zeros_like(voxel_sizes)

    def reset(self):
        self.ego_to_map = None
        self.initial_pose = None
        self.global_pose = None
        self.translation = torch.zeros_like(self.voxel_sizes)

    def forward(self, new_pose, current_map):
        prev_pose = self.global_pose
        self.global_pose = new_pose
        if self.initial_pose is None:
            prev_pose = new_pose
            self.initial_pose = new_pose
            # self.initial_pose = new_pose
            # return torch.eye(4).to(new_pose.device), current_map
        # Find the closest translation in voxels
        prev_to_initial = torch.matmul(torch.linalg.inv(self.initial_pose), prev_pose)
        prev_translation = prev_to_initial[:3, 3]
        prev_voxel = torch.round(prev_translation / self.voxel_sizes)

        new_to_initial = torch.matmul(torch.linalg.inv(self.initial_pose), new_pose)
        new_translation = new_to_initial[:3, 3]
        R = new_to_initial[:3, :3]
        new_voxel = torch.round(new_translation / self.voxel_sizes)

        self.translation = new_voxel * self.voxel_sizes
        voxel_translation = new_voxel - prev_voxel
        # Transform Map
        new_map = torch.zeros_like(current_map) + self.prior
        b = grid_limits(current_map, voxel_translation)
        new_map[b[0][0]:b[0][1], b[1][0]:b[1][1], b[2][0]:b[2][1], :] = \
            current_map[b[0][2]:b[0][3], b[1][2]:b[1][3], b[2][2]:b[2][3], :]
        # Transform from ego to map frame
        self.ego_to_map = torch.zeros_like(new_pose)
        self.ego_to_map[:3, 3] = new_translation - self.translation
        self.ego_to_map[:3, :3] = R
        self.ego_to_map[3, 3] = 1
        return self.ego_to_map, new_map


if __name__ == "__main__":
    world_transformer = TransformWorldStatic(torch.tensor([0.25, 0.25, 0.25]))
    grid = torch.ones((10, 10, 5))
    pose_1 = torch.eye(4)
    pose_1[0, 3] = 1.5
    pose_1[1, 3] = 0.7
    pose_1[2, 3] = 0.5
    pose_2 = torch.eye(4)
    pose_2[0, 3] = 1.5
    pose_2[1, 3] = 0.5
    pose_2[2, 3] = 0.5
    ego_to_map, grid = world_transformer(pose_1, grid)
    print(ego_to_map)
    print(grid[0, :, 0])
    ego_to_map, grid = world_transformer(pose_2, grid)
    print(ego_to_map)
    print(grid[0, :, 0])