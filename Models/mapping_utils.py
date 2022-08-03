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
                                         device=device, dtype=datatype, bias=False)
        self.ConvLayer.weight.requires_grad = False
        self.ConvLayer.weight[:, :, :, :, :] = weights.detach()[:, :, :, :, :]

        self.ConvLayer.eval()

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
        update = self.ConvLayer(update)
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

        # Group convolution layer
        self.ConvLayer = torch.nn.Conv3d(num_classes, num_classes, filter_size, padding="same", groups=num_classes,
                                         device=device, dtype=datatype, bias=False)
        self.ConvLayer.weight.requires_grad = False
        self.ConvLayer.weight[:, :, :, :, :] = weights.detach()[:, :, :, :, :]

    def reset_grid(self):
        self.global_map = None
        self.initial_pose = None
        self.translation_discretized = np.zeros(3)
        self.points_rotation = torch.eye(3, dtype=self.dtype, device=self.device)
        self.points_translation = torch.zeros(3, dtype=self.dtype, device=self.device)

    def inside_mask(self, min_bounds, max_bounds):
        inside = np.all((self.global_map[:, :3] >= min_bounds) & (self.global_map[:, :3] < max_bounds), axis=1)
        return inside

    def get_local_map(self, min_bound=None, max_bound=None):
        # Fetch local map from CPU (anything not seen is prior)
        local_map = self.initialize_grid()
        inside_mask = None
        if min_bound is None:
            min_bound = self.min_bound
        if max_bound is None:
            max_bound = self.max_bound
        local_min_bound = min_bound + torch.from_numpy(self.voxel_translation).to(self.device)
        local_max_bound = max_bound + torch.from_numpy(self.voxel_translation).to(self.device)
        if self.global_map is not None:
            inside_mask = self.inside_mask(local_min_bound.detach().cpu().numpy(), local_max_bound.detach().cpu().numpy())
            allocated_map = torch.tensor(self.global_map[inside_mask], device=self.device, dtype=self.dtype)
            grid_map = self.grid_ind(allocated_map, min_bound=local_min_bound, max_bound=local_max_bound)
            grid_indices = grid_map[:, :3].to(torch.long)
            local_map[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], :] = allocated_map[:, 3:]
        return local_map, local_min_bound, local_max_bound, inside_mask

    # Uses saved weights instead of generating a filter
    def update_map(self, semantic_preds):
        semantic_preds = semantic_preds.to(self.dtype)
        local_map, local_min_bound, local_max_bound, inside_mask = self.get_local_map()

        # Rotate the point cloud and translate to global frame
        global_pose = torch.from_numpy(self.global_pose).to(self.device)
        semantic_preds[:, :3] = torch.matmul(global_pose[:3, :3], semantic_preds[:, :3].T).T + global_pose[:3, 3]

        # Change to indices using our global frame bounds
        grid_pc = self.grid_ind(semantic_preds, min_bound=local_min_bound, max_bound=local_max_bound)

        # Update local map
        update = torch.zeros_like(local_map, requires_grad=False)

        continuous = False
        N, C = semantic_preds.shape
        if C == self.num_classes + 3:
            continuous = True

        update = self.add_to_update(update, grid_pc, continuous)

        # Apply BKI filters
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
        update = self.ConvLayer(update)
        new_update = torch.squeeze(update).permute(1, 2, 3, 0)

        # Find updated cells
        local_map = local_map + new_update
        updated_cells = (torch.mean(local_map, dim=3) > self.prior).view(-1)

        updated_centroids = self.centroids[updated_cells, :] + torch.from_numpy(self.voxel_translation).to(self.device)
        local_values = local_map.view(-1, self.num_classes)[updated_cells]
        new_cells = torch.cat((updated_centroids, local_values), dim=1)
        # If empty
        if self.global_map is None:
            self.global_map = new_cells.detach().cpu().numpy()
        else:
            # Replace local cells
            outside_mask = ~ inside_mask
            # Add new cells
            self.global_map = np.vstack((self.global_map[outside_mask, :], new_cells.detach().cpu().numpy()))
        return self.global_map

    # Propagate map given a transformation matrix
    def propagate(self, pose):
        self.global_pose = pose
        # Was just initialized
        if self.initial_pose is None:
            self.initial_pose = pose
        # Relative transformation between origin and current point
        relative_translation = pose[:3, 3] - self.initial_pose[:3, 3]
        # To select voxels from memory, find the nearest voxel
        voxel_sizes = self.voxel_sizes.detach().cpu().numpy()
        self.voxel_translation = np.round(relative_translation / voxel_sizes) * voxel_sizes
        self.nearest_voxel = self.initial_pose[:3, 3] + self.voxel_translation

    # Predict labels for points after propagating pose
    def label_points(self, points):
        points = torch.from_numpy(points).to(self.device)
        global_pose = torch.from_numpy(self.global_pose).to(self.device)
        points = torch.matmul(global_pose[:3, :3], points.T).T + global_pose[:3, 3]
        labels = torch.zeros((points.shape[0], self.num_classes), dtype=torch.float32, device=self.device)

        local_map, local_min_bound, local_max_bound, __ = self.get_local_map()

        local_mask = torch.all((points < local_max_bound) & (points >= local_min_bound), dim=1)

        local_points = points[local_mask]

        grid_inds = torch.floor((local_points - local_min_bound) / self.voxel_sizes)
        maxes = (self.grid_size - 1).view(1, 3)
        clipped_inds = torch.clamp(grid_inds, torch.zeros_like(maxes), maxes).to(torch.long)

        labels[local_mask, :] = local_map[clipped_inds[:, 0], clipped_inds[:, 1], clipped_inds[:, 2], :]
        labels[~local_mask, :] = self.prior

        # TODO: Add some sort of thresholding based on variance
        # TODO: Add calculation of expectation, variance
        predictions = torch.argmax(labels, dim=1)
        predictions[~local_mask] = 0
        return predictions, local_mask





