import pdb
import os

import torch
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F


class ConvBKI(torch.nn.Module):
    def __init__(self, grid_size, min_bound, max_bound, filter_size=3,
                 num_classes=21, prior=0.001, device="cpu", datatype=torch.float32,
                max_dist=0.5, kernel="sparse", per_class=False):
        '''
        Input:
            grid_size: (x, y, z) int32 array, number of voxels
            min_bound: (x, y, z) float32 array, lower bound on local map
            max_bound: (x, y, z) float32 array, upper bound on local map
            filter_size: int, dimension of the kernel on each axis (must be odd)
            num_classes: int, number of classes
            prior: float32, value of prior in map
            device: cpu or gpu
            max_dist: size of the kernel ell parameter
            kernel: kernel to choose
            per_class: whether to learn a different kernel for each class
        '''
        super().__init__()
        self.min_bound = min_bound.view(-1, 3).to(device)
        self.max_bound = max_bound.view(-1, 3).to(device)
        self.grid_size = grid_size
        self.dtype = datatype
        self.prior = prior

        self.kernel = kernel
        self.device = device
        self.num_classes = num_classes
        self.per_class = per_class
        
        self.voxel_sizes = (self.max_bound.view(-1) - self.min_bound.view(-1)) / self.grid_size.to(self.device)
        
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.max_dist = max_dist
        self.filter_size = torch.tensor(filter_size, dtype=torch.long, requires_grad=False, device=self.device)

        self.initialize_kernel()
        
        [xs, ys, zs] = [(max_bound[i]-min_bound[i])/(2*grid_size[i]) + 
                        torch.linspace(min_bound[i], max_bound[i], device=device, steps=grid_size[i]+1)[:-1] 
                        for i in range(3)]

        self.centroids = torch.cartesian_prod(xs, ys, zs).to(device)

    def initialize_kernel(self):
        # Initialize with sparse kernel
        assert(self.filter_size % 2 == 1)

        # Parameters
        if self.kernel == "sparse":
            self.sigma = torch.tensor(1.0, device=self.device) # Kernel must map to 0 to 1
            if self.per_class:
                self.ell = torch.nn.Parameter(torch.tensor([self.max_dist] * self.num_classes, device=self.device))
            else:
                self.ell = torch.nn.Parameter(torch.tensor(self.max_dist, device=self.device, dtype=self.dtype))

        # Distances
        middle_ind = torch.floor(self.filter_size / 2)
        self.kernel_dists = torch.zeros([1, 1, self.filter_size, self.filter_size, self.filter_size], device=self.device)
        for x_ind in range(self.filter_size):
            for y_ind in range(self.filter_size):
                for z_ind in range(self.filter_size):
                    x_dist = torch.abs(x_ind - middle_ind) * self.voxel_sizes[0]
                    y_dist = torch.abs(y_ind - middle_ind) * self.voxel_sizes[1]
                    z_dist = torch.abs(z_ind - middle_ind) * self.voxel_sizes[2]
                    total_dist = torch.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
                    self.kernel_dists[0, 0, x_ind, y_ind, z_ind] = total_dist

    def calculate_kernel(self, d, i=0):
        if self.kernel == "sparse":
            if self.per_class:
                kernel_val = self.sigma * (
                        (1.0 / 3) * (2 + torch.cos(2 * self.pi * d / self.ell[i])) * (1 - d / self.ell[i]) +
                        1.0 / (2 * self.pi) * torch.sin(2 * self.pi * d / self.ell[i])
                )
                kernel_val[d >= self.ell[i]] = 0
                return torch.clamp(kernel_val, min=0.0, max=1.0)
            else:
                kernel_val = self.sigma * (
                        (1.0/3)*(2 + torch.cos(2 * self.pi * d/self.ell))*(1 - d/self.ell) +
                                 1.0/(2*self.pi) * torch.sin(2 * self.pi * d / self.ell)
                                 )
                kernel_val[d >= self.ell] = 0
                return torch.clamp(kernel_val, min=0.0, max=1.0)
        return None

    def initialize_grid(self):
        return torch.zeros(self.grid_size[0], self.grid_size[1], self.grid_size[2], 
                           self.num_classes, device=self.device, requires_grad=True,
                           dtype=self.dtype) + self.prior
    
    def grid_ind(self, input_pc):
        '''
        Input:
            input_xyz: N * (x, y, z, c) float32 array, point cloud
        Output:
            grid_inds: N' * (x, y, z, c) int32 array, point cloud mapped to voxels
        '''
        input_xyz   = input_pc[:, :3]
        labels      = input_pc[:, 3].view(-1, 1)
        
        valid_input_mask = torch.all((input_xyz < self.max_bound) & (input_xyz >= self.min_bound), axis=1)
        
        valid_xyz = input_xyz[valid_input_mask]
        valid_labels = labels[valid_input_mask]
        
        grid_inds = torch.floor((valid_xyz - self.min_bound) / self.voxel_sizes)
        maxes = (self.grid_size - 1).view(1, 3)
        clipped_inds = torch.clamp(grid_inds, torch.zeros_like(maxes), maxes)
        
        return torch.hstack( (clipped_inds, valid_labels) )

    def get_filters(self):
        filters = torch.zeros([self.num_classes, self.num_classes, self.filter_size, self.filter_size, self.filter_size],
                              device=self.device, dtype=self.dtype)
        for temp_class in range(self.num_classes):
            if self.per_class:
                filters[temp_class, temp_class, :, :, :] = self.calculate_kernel(self.kernel_dists, i=temp_class)
            else:
                filters[temp_class, temp_class, :, :, :] = self.calculate_kernel(self.kernel_dists)
        return filters

    def forward(self, current_map, point_cloud):
        '''
        Input:
            current_map: (x, y, z, c) float32 array, prior dirichlet distribution over map
            point_cloud: N * (x, y, z, c) float32 array, semantically labeled points
        Output:
            updated_map: (x, y, z, c) float32 array, posterior dirichlet distribution over map
        '''
        # Assume map and point cloud are already aligned
        X, Y, Z, C = current_map.shape
        update = torch.zeros_like(current_map, requires_grad=False)
        
        # 1: Discretize
        grid_pc = self.grid_ind(point_cloud).to(torch.long)
       
        unique_inds, counts = torch.unique(grid_pc, return_counts=True, dim=0)
        counts = counts.type(torch.long)

        grid_indices = [unique_inds[:, i] for i in range(grid_pc.shape[1])]
        update[grid_indices] = update[grid_indices] + counts
        
        # 2: Apply BKI filters
        filters = self.get_filters()
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
        update = F.conv3d(update, filters, padding="same")
        new_update = torch.squeeze(update).permute(1, 2, 3, 0)

        return current_map + new_update