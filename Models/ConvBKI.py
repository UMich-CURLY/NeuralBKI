import pdb
import os

import torch
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F


class ConvBKI(torch.nn.Module):
    def __init__(self, grid_size, min_bound, max_bound, filter_size=3,
                 num_classes=21, prior=0.001, device="cpu", datatype=torch.float32,
                max_dist=0.5, kernel="sparse", per_class=False, compound=False):
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
        self.compound = compound
        
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
        self.sigma = torch.tensor(1.0, device=self.device)  # Kernel must map to 0 to 1

        # Parameters
        if self.kernel == "sparse":
            if self.compound:
                if self.per_class:
                    self.ell_h = torch.nn.Parameter(torch.tensor([self.max_dist] * self.num_classes, device=self.device))
                    self.ell_z = torch.nn.Parameter(torch.tensor([self.max_dist] * self.num_classes, device=self.device))
                    # self.ell_h = torch.nn.Parameter(0.2 + self.max_dist*torch.rand(self.num_classes, device=self.device))
                    # self.ell_z = torch.nn.Parameter(0.2 + self.max_dist*torch.rand(self.num_classes, device=self.device))
                else:
                    self.ell_h = torch.nn.Parameter(torch.tensor(self.max_dist, device=self.device, dtype=self.dtype))
                    self.ell_z = torch.nn.Parameter(torch.tensor(self.max_dist, device=self.device, dtype=self.dtype))
            else:
                if self.per_class:
                    self.ell = torch.nn.Parameter(torch.tensor([self.max_dist] * self.num_classes, device=self.device))
                    # self.ell = torch.nn.Parameter(2*self.max_dist*torch.rand(self.num_classes, device=self.device))
                else:
                    self.ell = torch.nn.Parameter(torch.tensor(self.max_dist, device=self.device, dtype=self.dtype))

        # Distances
        middle_ind = torch.floor(self.filter_size / 2)
        if self.compound:
            self.kernel_dists_h = torch.zeros([1, 1, self.filter_size, self.filter_size, self.filter_size],
                                            device=self.device)
            self.kernel_dists_z = torch.zeros([1, 1, self.filter_size, self.filter_size, self.filter_size],
                                            device=self.device)
        else:
            self.kernel_dists = torch.zeros([1, 1, self.filter_size, self.filter_size, self.filter_size],
                                            device=self.device)
        for x_ind in range(self.filter_size):
            for y_ind in range(self.filter_size):
                for z_ind in range(self.filter_size):
                    x_dist = torch.abs(x_ind - middle_ind) * self.voxel_sizes[0]
                    y_dist = torch.abs(y_ind - middle_ind) * self.voxel_sizes[1]
                    z_dist = torch.abs(z_ind - middle_ind) * self.voxel_sizes[2]
                    if self.compound:
                        horiz_dist = torch.sqrt(x_dist ** 2 + y_dist ** 2)
                        vert_dist = torch.sqrt(z_dist ** 2)
                        self.kernel_dists_h[0, 0, x_ind, y_ind, z_ind] = horiz_dist
                        self.kernel_dists_z[0, 0, x_ind, y_ind, z_ind] = vert_dist
                    else:
                        total_dist = torch.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
                        self.kernel_dists[0, 0, x_ind, y_ind, z_ind] = total_dist

    def sparse_kernel(self, d, ell, sigma):
        kernel_val = sigma * ((1.0/3)*(2 + torch.cos(2 * self.pi * d/ell))*(1 - d/ell) +
                              1.0/(2*self.pi) * torch.sin(2 * self.pi * d / ell))
        kernel_val[d >= ell] = 0
        return torch.clamp(kernel_val, min=0.0, max=1.0)

    def calculate_kernel(self, i=0):
        kernel_val = None
        if self.kernel == "sparse":
            if self.per_class:
                if self.compound:
                    kernel_val = self.sparse_kernel(self.kernel_dists_z, self.ell_z[i], self.sigma) * \
                                 self.sparse_kernel(self.kernel_dists_h, self.ell_h[i], self.sigma)
                else:
                    kernel_val = self.sparse_kernel(self.kernel_dists, self.ell[i], self.sigma)
            else:
                kernel_val = self.sparse_kernel(self.kernel_dists, self.ell, self.sigma)
        return kernel_val

    def initialize_grid(self):
        return torch.zeros(self.grid_size[0], self.grid_size[1], self.grid_size[2], 
                           self.num_classes, device=self.device, requires_grad=True,
                           dtype=self.dtype) + self.prior
    
    def grid_ind(self, input_pc, min_bound=None, max_bound=None):
        '''
        Input:
            input_xyz: N * (x, y, z, c) float32 array, point cloud
        Output:
            grid_inds: N' * (x, y, z, c) int32 array, point cloud mapped to voxels
        '''
        if min_bound is None:
            min_bound = self.min_bound
        if max_bound is None:
            max_bound = self.max_bound
        input_xyz   = input_pc[:, :3]
        labels      = input_pc[:, 3:]
        
        valid_input_mask = torch.all((input_xyz < max_bound) & (input_xyz >= min_bound), axis=1)
        
        valid_xyz = input_xyz[valid_input_mask]
        valid_labels = labels[valid_input_mask]
        
        grid_inds = torch.floor((valid_xyz - min_bound) / self.voxel_sizes)
        maxes = (self.grid_size - 1).view(1, 3)
        clipped_inds = torch.clamp(grid_inds, torch.zeros_like(maxes), maxes)
        
        return torch.hstack( (clipped_inds, valid_labels) )

    def get_filters(self):
        filters = torch.zeros([self.num_classes, 1, self.filter_size, self.filter_size, self.filter_size],
                              device=self.device, dtype=self.dtype)
        for temp_class in range(self.num_classes):
            if self.per_class:
                filters[temp_class, 0, :, :, :] = self.calculate_kernel(i=temp_class)
            else:
                filters[temp_class, 0, :, :, :] = self.calculate_kernel()
        return filters

    def add_to_update(self, update, grid_pc, continuous=False):
        if continuous:
            # Solution inspired by https://github.com/facebookresearch/SparseConvNet/blob/main/sparseconvnet/utils.py
            xyz = grid_pc[:, :3]
            feat = grid_pc[:, 3:]
            xyz, inv, counts = torch.unique(xyz, dim=0, return_inverse=True, return_counts=True)
            feat_out = torch.zeros(xyz.size(0), feat.size(1), dtype=torch.float32, device=self.device)
            feat_out.index_add_(0, inv, feat)
            grid_ind = xyz.to(torch.long)
            update[grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]] = feat_out

        else:
            unique_inds, counts = torch.unique(grid_pc.to(torch.long), return_counts=True, dim=0)
            counts = counts.type(torch.long)
            grid_indices = [unique_inds[:, i] for i in range(grid_pc.shape[1])]
            update[grid_indices] = update[grid_indices] + counts
        return update

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

        N, C = point_cloud.shape
        continuous = False
        if C == self.num_classes + 3:
            continuous = True
        
        # 1: Discretize
        grid_pc = self.grid_ind(point_cloud)
        update = self.add_to_update(update, grid_pc, continuous)
        
        # 2: Apply BKI filters
        filters = self.get_filters()
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
        update = F.conv3d(update, filters, padding="same", groups=self.num_classes)
        new_update = torch.squeeze(update).permute(1, 2, 3, 0)

        return current_map + new_update