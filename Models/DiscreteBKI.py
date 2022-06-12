import pdb
import os

import torch
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
from Models.BKIConvFilter import BKIConvFilter

class DiscreteBKI(torch.nn.Module):
    def __init__(self, grid_size, min_bound, max_bound, filter_size=3,
                 num_classes=21, prior=0.001, device="cpu", datatype=torch.float32,
                max_dist=0.5):
        '''
        Input:
            grid_size: (x, y, z) int32 array, number of voxels
            min_bound: (x, y, z) float32 array, lower bound on local map
            max_bound: (x, y, z) float32 array, upper bound on local map
        '''
        super().__init__()
        self.min_bound = min_bound.view(-1, 3).to(device)
        self.max_bound = max_bound.view(-1, 3).to(device)
        self.grid_size = grid_size
        self.dtype = datatype
        self.prior = prior

        self.device = device
        self.num_classes = num_classes
        
        self.voxel_sizes = (self.max_bound.view(-1) - self.min_bound.view(-1)) / self.grid_size.to(self.device)
        
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.max_dist = max_dist
        self.filter_size = torch.tensor(filter_size, dtype=torch.long, requires_grad=False, device=self.device)

        self.bki_conv_filter = BKIConvFilter.apply
        self.initialize_kernel()
        
        [xs, ys, zs] = [(max_bound[i]-min_bound[i])/(2*grid_size[i]) + 
                        torch.linspace(min_bound[i], max_bound[i], device=device, steps=grid_size[i]+1)[:-1] 
                        for i in range(3)]

        self.centroids = torch.cartesian_prod(xs, ys, zs).to(device)
    
    def initialize_kernel(self):
        # Initialize with sparse kernel
        weights = []
        assert(self.filter_size % 2 == 1)
        middle_ind = torch.floor(self.filter_size / 2)
        
        # self.sigma = torch.nn.Parameter(torch.tensor(1.0)) # Kernel must map to 0 to 1
        # self.ell = torch.nn.Parameter(torch.tensor(self.max_dist)) # Max distance to consider
        self.sigma = torch.tensor(1.0) # Kernel must map to 0 to 1
        self.ell = torch.tensor(self.max_dist) # Max distance to consider
        
        for x_ind in range(self.filter_size):
            for y_ind in range(self.filter_size):
                for z_ind in range(self.filter_size):
                    x_dist = torch.abs(x_ind - middle_ind) * self.voxel_sizes[0]
                    y_dist = torch.abs(y_ind - middle_ind) * self.voxel_sizes[1]
                    z_dist = torch.abs(z_ind - middle_ind) * self.voxel_sizes[2]
                    total_dist = torch.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
                    kernel_value = self.calculate_kernel(total_dist)
                    # Edge case: middle
                    if total_dist == 0:
                        weights.append(1.0)
                    else:
                        weight = self.inverse_sigmoid(kernel_value)
                        weights.append(torch.nn.Parameter(weight))
        weights = torch.tensor(weights, dtype=self.dtype, device=self.device).view(
            1, 1, self.filter_size, self.filter_size, self.filter_size)
        # pdb.set_trace()

        # Random Initialization
        torch.nn.init.normal_(weights, mean=0, std=0.1)
        middle_ind = middle_ind.long()
        weights[:, :, middle_ind, middle_ind, middle_ind] = 1.0
        self.weights = torch.nn.Parameter(weights)

    def inverse_sigmoid(self, x):
        return -torch.log((1 / (x + 1e-8)) - 1)
        # return -torch.log( (1-x) /  (x+1e-8) )
            
    def calculate_kernel(self, d):
        if d > self.max_dist:
            return torch.tensor(0.0, device=self.device)
        if d == 0:
            return 1
        return self.sigma * ( 
                (1.0/3)*(2 + torch.cos(2 * self.pi * d/self.ell))*(1 - d/self.ell) +
                         1.0/(2*self.pi) * torch.sin(2 * self.pi * d / self.ell)
                         )
            
            
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
        # filters = torch.sigmoid(
        #     self.weights
        # )
        mid = torch.floor(self.filter_size / 2).to(torch.long)
        filters = self.bki_conv_filter(self.weights, mid)

        # filters[0, 0, mid, mid, mid] = 1
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 1)
        update = F.conv3d(update, filters, padding="same")
        new_update = torch.squeeze(update).permute(1, 2, 3, 0)
        
        return current_map + new_update
    
    # def propagate(self, current_map, transformation)