import torch

# BKINet consists of two components:
# 1) A pre-trained semantic segmentation model
# 2) A pre-trained ConvBKI layer
# This module is intended for ROS integration


class BKINet(torch.nn.Module):
    def __init__(self, grid_size, min_bound, max_bound, weights, filter_size, segmentation_net,
                 num_classes=21, prior=0.001, device="cpu", datatype=torch.float32):
        super().__init__()
        self.segmentation_net = segmentation_net

        self.min_bound = min_bound.view(-1, 3).to(device)
        self.max_bound = max_bound.view(-1, 3).to(device)
        self.grid_size = grid_size
        self.dtype = datatype

    def grid_ind(self, input_pc):
        '''
        Input:
            input_xyz: N * (x, y, z, c) float32 array, point cloud
        Output:
            grid_inds: N' * (x, y, z, c) int32 array, point cloud mapped to voxels
        '''
        input_xyz = input_pc[:, :3]
        labels = input_pc[:, 3].view(-1, 1)

        valid_input_mask = torch.all((input_xyz < self.max_bound) & (input_xyz >= self.min_bound), axis=1)

        valid_xyz = input_xyz[valid_input_mask]
        valid_labels = labels[valid_input_mask]

        grid_inds = torch.floor((valid_xyz - self.min_bound) / self.voxel_sizes)
        maxes = (self.grid_size - 1).view(1, 3)
        clipped_inds = torch.clamp(grid_inds, torch.zeros_like(maxes), maxes)

        return torch.hstack((clipped_inds, valid_labels))

