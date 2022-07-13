# This file contains classes for local and global offline mapping (not running semantic prediction)
import torch
import torch.nn.functional as F
import numpy as np
from Models.ConvBKI import ConvBKI


# Naive implementation, we can implement ourselves to make this much more efficient
# Our conv only considers same-class, so this is K^2 less efficient
# https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
# Also, we could improve by only processing areas with measurements (sparse convolution)
# https://medium.com/geekculture/3d-sparse-sabmanifold-convolutions-eaa427b3a196
# https://github.com/traveller59/spconv/blob/master/docs/USAGE.md
# current_map: (x, y, z, c) float32 array, prior dirichlet distribution over map


# TODO: Trilinear interpolation
class LocalMap(ConvBKI):
    def __init__(self, grid_size, min_bound, max_bound, weights, filter_size, num_classes=21, prior=0.001, device="cpu",
                 datatype=torch.float32):
        super().__init__(grid_size, min_bound, max_bound, filter_size=filter_size,
                 num_classes=num_classes, prior=prior, device=device, datatype=datatype)

        self.weights = weights
        self.reset_grid()

        # Group convolution layer
        self.ConvLayer = torch.nn.Conv3d(num_classes, num_classes, filter_size, padding="same", groups=num_classes,
                                         device=device, dtype=datatype)
        self.ConvLayer.weight.requires_grad = False
        for i in range(num_classes):
            self.ConvLayer.weight[i, 0, :, :, :] = weights[i, i, :, :, :].detach()

    def reset_grid(self):
        self.current_map = self.initialize_grid()
        self.pose = None

    # Uses saved weights instead of generating a filter
    def update_map(self, semantic_preds):
        update = torch.zeros_like(self.current_map, requires_grad=False)

        # 1: Discretize
        grid_pc = self.grid_ind(semantic_preds).to(torch.long)

        unique_inds, counts = torch.unique(grid_pc, return_counts=True, dim=0)
        counts = counts.type(torch.long)

        grid_indices = [unique_inds[:, i] for i in range(grid_pc.shape[1])]
        update[grid_indices] = update[grid_indices] + counts

        # 2: Apply BKI filters
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
        update = F.conv3d(update, self.weights, padding="same")
        temp_update = self.ConvLayer(update)
        print("Difference:", torch.sum(abs(update - temp_update)))
        new_update = torch.squeeze(update).permute(1, 2, 3, 0)

        self.current_map = self.current_map + new_update

    # Propagate map given a transformation matrix
    def propagate(self, pose):
        pose = torch.from_numpy(pose).to(self.device)
        # Was just initialized
        if self.pose is None:
            self.pose = pose
            return
        # Calculate query locations
        relative_pose = torch.matmul(torch.linalg.inv(pose), self.pose)
        orig_centroids = torch.clone(self.centroids.detach())
        new_centroids = torch.cat((orig_centroids, torch.ones_like(orig_centroids)[:, 0:1]), dim=1)
        new_centroids = torch.matmul(new_centroids, relative_pose)
        # Infer at new position (trilinear interpolation)


# Save grid in CPU memory, load to GPU when needed for update step
# Voxels are stored in a matrix [X | Y | Z | C_0 | ... C_N] where C is semantic class
class GlobalMap(ConvBKI):
    def __init__(self, grid_size, min_bound, max_bound, weights, filter_size, num_classes=21, prior=0.001, device="cpu",
                 datatype=torch.float32):
        super().__init__(grid_size, min_bound, max_bound, filter_size=filter_size,
                 num_classes=num_classes, prior=prior, device=device, datatype=datatype)

        self.weights = weights
        self.reset_grid()

    def reset_grid(self):
        self.global_map = None
        self.initial_pose = None

    # Uses saved weights instead of generating a filter
    def update_map(self, semantic_preds):
        # Fetch local map from CPU (anything not seen is prior)
        return

        # Update local map
        update = torch.zeros_like(self.current_map, requires_grad=False)

        # Discretize
        grid_pc = self.grid_ind(semantic_preds).to(torch.long)

        unique_inds, counts = torch.unique(grid_pc, return_counts=True, dim=0)
        counts = counts.type(torch.long)

        grid_indices = [unique_inds[:, i] for i in range(grid_pc.shape[1])]
        update[grid_indices] = update[grid_indices] + counts

        # Apply BKI filters
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
        update = F.conv3d(update, self.weights, padding="same")

        new_update = torch.squeeze(update).permute(1, 2, 3, 0)

        # Update local indices

        self.current_map = self.current_map + new_update

    # Propagate map given a transformation matrix
    def propagate(self, pose):
        pose = torch.from_numpy(pose).to(self.device)
        # Was just initialized
        if self.initial_pose is None:
            self.initial_pose = pose
            return
        # GLobal frame pose
        global_pose = torch.matmul(torch.linalg.inv(pose), self.initial_pose)
        translation = global_pose[:3, 3]
        # To select voxels from memory
        translation_discretized = torch.round(offset / self.voxel_sizes) * self.voxel_sizes
        # Transformation on point cloud to account for discretization error
        error = translation - translation_discretized
