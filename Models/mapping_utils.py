# This file contains classes for local and global offline mapping (not running semantic prediction)
import torch
import torch.nn.functional as F
import numpy as np
from Models.ConvBKI import ConvBKI
import copy

# Naive implementation, we can implement ourselves to make this much more efficient
# Our conv only considers same-class, so this is K^2 less efficient
# https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
# Also, we could improve by only processing areas with measurements (sparse convolution)
# https://medium.com/geekculture/3d-sparse-sabmanifold-convolutions-eaa427b3a196
# https://github.com/traveller59/spconv/blob/master/docs/USAGE.md
# current_map: (x, y, z, c) float32 array, prior dirichlet distribution over map


# TODO: Trilinear interpolation
class LocalMapNotDone(ConvBKI):
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

    # Uses saved weights instead of generating a filter
    def update_map(self, semantic_preds):
        # Fetch local map from CPU (anything not seen is prior)
        local_map = self.initialize_grid() # H x W x D x Classes

        if self.global_map is not None:
            allocated_map = self.global_map
            allocated_map[:, :3] = allocated_map[:, :3] - self.translation_discretized
            allocated_map = self.grid_ind(torch.tensor(allocated_map, device=self.device, dtype=self.dtype))
            grid_indices = allocated_map[:, :3].to(torch.long)
            local_map[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], :] = allocated_map[:, 3:]


        # if self.global_map is not None:
        #     allocated_map = torch.tensor(self.global_map)
        #     allocated_map[:, :3] = allocated_map[:, :3] - self.translation_discretized
        #     # check for updated cells
        #     updated = (torch.mean(allocated_map[:,3:], dim=1) > self.prior).view(-1)
        #     labels = torch.argmax(allocated_map[:,3:] / torch.sum(allocated_map[:,3:], axis=-1, keepdims=True), axis=-1)
        #     labels[~updated] = 0
        #     labels = labels.unsqueeze(1)
        #     allocated_map = torch.hstack((allocated_map[:,:3], labels)) # H X W x D x 4 

        #     allocated_map = self.grid_ind(torch.tensor(allocated_map, device=self.device, dtype=self.dtype))
        #     grid_pc = self.grid_ind(torch.tensor(allocated_map, device=self.device, dtype=self.dtype)).to(torch.long)
        #     grid_indices = allocated_map[:, :3].to(torch.long)
        #     local_map[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], allocated_map[:,3].to(torch.long)] += 1


        semantic_preds[:, :3] = torch.matmul(self.points_rotation, semantic_preds[:, :3].T).T + self.points_translation

        # Update local map
        update = torch.zeros_like(local_map, requires_grad=False)
        # Discretize
        grid_pc = self.grid_ind(semantic_preds).to(torch.long)

        unique_inds, counts = torch.unique(grid_pc, return_counts=True, dim=0)
        counts = counts.type(torch.long)
        grid_indices = [unique_inds[:, i] for i in range(grid_pc.shape[1])]
        update[grid_indices] = update[grid_indices] + counts

        # Apply BKI filters
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
        #update = self.ConvLayer(update)
        new_update = torch.squeeze(update).permute(1, 2, 3, 0)

        # Find updated cells
        local_map = local_map + new_update
        updated_cells = (torch.mean(local_map, dim=3) > self.prior).view(-1)
        updated_centroids = self.centroids[updated_cells, :] + torch.tensor(self.translation_discretized, device=self.device)
        local_values = local_map.view(-1, self.num_classes)[updated_cells]
        new_cells = torch.cat((updated_centroids, local_values), dim=1)

        # If empty
        if self.global_map is None:
            self.global_map = new_cells.detach().cpu().numpy()
        else:
            outside_mask = np.any((self.global_map[:, :3] - self.translation_discretized >= self.max_bound.cpu().numpy()) |
                                (self.global_map[:, :3] - self.translation_discretized < self.min_bound.cpu().numpy()), axis=1)
            # print(np.sum(outside_mask))
            # Add new cells
            self.global_map = np.vstack((self.global_map[outside_mask, :], new_cells.detach().cpu().numpy()))
        return self.global_map

    # Propagate map given a transformation matrix
    def propagate(self, pose, prior_pose, Tr):
        pose = torch.from_numpy(pose).to(self.device)
        Tr = torch.from_numpy(Tr).to(self.device)

        # Was just initialized
        if self.initial_pose is None:
            self.initial_pose = pose
            self.relative_pose = prior_pose
            return

        # GLobal frame transformation (to get from current frame to global frame)
        #global_pose = torch.matmul(torch.linalg.inv(pose), self.initial_pose)
        global_pose = torch.matmul(torch.linalg.inv(Tr), torch.matmul(pose, Tr))
        translation = global_pose[:3, 3]
        # To select voxels from memory
        translation_discretized = torch.round(translation / self.voxel_sizes) * self.voxel_sizes
        # Transformation on point cloud to account for discretization error
        self.points_translation = translation - translation_discretized
        self.points_rotation = global_pose[:3, :3]
        self.translation_discretized = translation_discretized.detach().cpu().numpy()


class GlobalMap2(ConvBKI):
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
        self.relative_pose = None
        self.translation_discretized = np.zeros(3)
        self.points_rotation = torch.eye(3, dtype=self.dtype, device=self.device)
        self.points_translation = torch.zeros(3, dtype=self.dtype, device=self.device)

    # Uses saved weights instead of generating a filter
    def update_map(self, semantic_preds):
        # Fetch local map from CPU (anything not seen is prior)
        local_map = self.initialize_grid()

        # if self.global_map is not None:
        #     allocated_map = torch.tensor(self.global_map, device=self.device).float()
        #     # allocated_map[:, :3] = allocated_map[:, :3] - torch.tensor(self.translation_discretized, device=self.device)
        #     # check for updated cells
        #     allocated_map[:,:3] = torch.matmul(self.relative_pose[:3,:3], allocated_map[:,:3].T).T + self.relative_pose[:3,3]
        #     updated = (torch.mean(allocated_map[:,3:], dim=1) > self.prior).view(-1)
        #     labels = torch.argmax(allocated_map[:,3:] / torch.sum(allocated_map[:,3:], axis=-1, keepdims=True), axis=-1)
        #     labels[~updated] = 0
        #     labels = labels.unsqueeze(1)
        #     allocated_map = torch.hstack((allocated_map[:,:3], labels)) # H X W x D x 4 

        #     allocated_map = self.grid_ind(torch.tensor(allocated_map, device=self.device, dtype=self.dtype))
        #     grid_pc = self.grid_ind(torch.tensor(allocated_map, device=self.device, dtype=self.dtype)).to(torch.long)
        #     grid_indices = allocated_map[:, :3].to(torch.long)
        #     local_map[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], allocated_map[:,3].to(torch.long)] += 1

        # Update local map
        update = torch.zeros_like(local_map, requires_grad=False)

        # Discretize
        grid_pc = self.grid_ind(semantic_preds).to(torch.long)
        unique_inds, counts = torch.unique(grid_pc, return_counts=True, dim=0)
        counts = counts.type(torch.long)
        grid_indices = [unique_inds[:, i] for i in range(grid_pc.shape[1])]
        update[grid_indices] = update[grid_indices] + counts

        # Apply BKI filters
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
        # update = self.ConvLayer(update)
        new_update = torch.squeeze(update).permute(1, 2, 3, 0)
        # Find updated cells
        local_map = local_map + new_update

        # Transform local map centroids, max, minboubnd to global
        centroids = self.centroids
        centroids = torch.matmul(torch.tensor(self.points_rotation, device=self.device), centroids.T).T + torch.tensor(self.translation_discretized, device=self.device)
        min_bound = (torch.matmul(torch.tensor(self.points_rotation, device=self.device), self.min_bound.T).T + torch.tensor(self.translation_discretized, device=self.device)).float().detach().cpu().numpy()
        max_bound = (torch.matmul(torch.tensor(self.points_rotation, device=self.device), self.max_bound.T).T + torch.tensor(self.translation_discretized, device=self.device)).float().detach().cpu().numpy()

        global_map_t = torch.cat((centroids, local_map.reshape(-1,local_map.shape[-1])),dim=1)
        updated_cells = (torch.mean(local_map, dim=3) > self.prior).view(-1).detach().cpu().numpy()
        global_map_t = global_map_t[updated_cells]
        
        # If empty
        if self.global_map is None:
            self.global_map = global_map_t.detach().cpu().numpy()
        else:
            outside_mask = np.any((self.global_map[:, :3]  >= max_bound) |
                (self.global_map[:, :3] < min_bound), axis=1)
            self.global_map = np.vstack((self.global_map[outside_mask], global_map_t.detach().cpu().numpy()))
            
        return self.global_map

    # Propagate map given a transformation matrix
    def propagate(self, pose, prior_pose, Tr):
        pose = torch.from_numpy(pose).to(self.device)
        Tr = torch.from_numpy(Tr).to(self.device)
        prior_pose = torch.from_numpy(prior_pose).to(self.device)

        pose = torch.matmul(torch.linalg.inv(Tr), torch.matmul(pose, Tr))
        prior_pose = torch.matmul(torch.linalg.inv(Tr), torch.matmul(prior_pose, Tr))
        # Was just initialized
        if self.initial_pose is None:
            self.initial_pose = pose
            self.relative_pose = prior_pose
            return

        # GLobal frame transformation (to get from current frame to global frame)
        global_pose = pose 
        self.relative_pose = torch.matmul(torch.linalg.inv(pose), prior_pose)
        translation = global_pose[:3, 3]
        # To select voxels from memory
        #translation_discretized = torch.round(translation / self.voxel_sizes) * self.voxel_sizes
        # Transformation on point cloud to account for discretization error
        #self.points_translation = translation - translation_discretized
        self.points_rotation = global_pose[:3, :3]
        # self.translation_discretized = translation_discretized.detach().cpu().numpy()
        # self.translation_discretized = translation
        # semantikitti poses are in different orders
        self.translation_discretized = translation


class LocalMap2(ConvBKI):
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
        self.relative_pose = None
        self.translation_discretized = np.zeros(3)
        self.points_rotation = torch.eye(3, dtype=self.dtype, device=self.device)
        self.points_translation = torch.zeros(3, dtype=self.dtype, device=self.device)

    # Uses saved weights instead of generating a filter
    def update_map(self, semantic_preds):
        # Fetch local map from CPU (anything not seen is prior)
        local_map = self.initialize_grid()

        # Update local map
        update = torch.zeros_like(local_map, requires_grad=False)

        # Discretize
        grid_pc = self.grid_ind(semantic_preds).to(torch.long)

        unique_inds, counts = torch.unique(grid_pc, return_counts=True, dim=0)
        counts = counts.type(torch.long)
        print(unique_inds.shape, grid_pc.shape)
        grid_indices = [unique_inds[:, i] for i in range(grid_pc.shape[1])]
        update[grid_indices] = update[grid_indices] + counts

        # Apply BKI filters
        update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
        #update = self.ConvLayer(update)
        new_update = torch.squeeze(update).permute(1, 2, 3, 0)

        # Find updated cells
        local_map = local_map + new_update
        # Transform local map centroids to global
        centroids = self.centroids

        global_map_t = torch.cat((centroids, local_map.reshape(-1,local_map.shape[-1])),dim=1)
        updated_cells = (torch.mean(local_map, dim=3) > self.prior).view(-1).detach().cpu().numpy()
        global_map_t = global_map_t[updated_cells]
        self.global_map = global_map_t.detach().cpu().numpy()

        return self.global_map

    # Propagate map given a transformation matrix
    def propagate(self, pose, prior_pose, Tr):
        pose = torch.from_numpy(pose).to(self.device)
        Tr = torch.from_numpy(Tr).to(self.device)
        prior_pose = torch.from_numpy(prior_pose).to(self.device)

        pose = torch.matmul(torch.linalg.inv(Tr), torch.matmul(pose, Tr))
        prior_pose = torch.matmul(torch.linalg.inv(Tr), torch.matmul(prior_pose, Tr))
        # Was just initialized
        if self.initial_pose is None:
            self.initial_pose = pose
            self.relative_pose = prior_pose
            return

        # GLobal frame transformation (to get from current frame to global frame)
        #global_pose = torch.matmul(torch.linalg.inv(pose), self.initial_pose)
        global_pose = torch.matmul(torch.linalg.inv(Tr), torch.matmul(pose, Tr))
        self.relative_pose = torch.matmul(torch.linalg.inv(pose), prior_pose)
        translation = global_pose[:3, 3]
        # To select voxels from memory
        #translation_discretized = torch.round(translation / self.voxel_sizes) * self.voxel_sizes
        # Transformation on point cloud to account for discretization error
        #self.points_translation = translation - translation_discretized
        self.points_rotation = global_pose[:3, :3]

        self.translation_discretized = translation


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

    def reset_grid(self):
        self.current_map = self.initialize_grid()
        self.initial_pose = None
        self.pose = None
        self.prior_map = None
        self.global_map = None
    def generate_grids(self):
        x = torch.linspace(self.min_bound[0][0], self.max_bound[0][0], steps=self.grid_size[0]).to(device=self.current_map.device)
        y = torch.linspace(self.min_bound[0][1], self.max_bound[0][1], steps=self.grid_size[1]).to(device=self.current_map.device)
        z = torch.linspace(self.min_bound[0][2], self.max_bound[0][2], steps=self.grid_size[2]).to(device=self.current_map.device)
        xv, yv, zv = torch.meshgrid((x,y,z), indexing="ij")
        xv = xv.reshape(-1)
        yv = yv.reshape(-1)
        zv = zv.reshape(-1)
        coords = torch.stack((xv, yv, zv), axis=1) 
        return coords

    # Uses saved weights instead of generating a filter
    def update_map(self, semantic_preds):
        update = torch.zeros_like(self.current_map, requires_grad=False)

        # 1: Discretize
        if self.prior_map is None:
            coords = self.generate_grids()
            self.prior_map = self.current_map
            self.global_map = torch.cat((coords.reshape(-1, 3), self.current_map.reshape(-1,self.current_map.shape[-1])),dim=1)

        else:
            coords = self.generate_grids()
            ## current
            
            grid_pc = self.grid_ind(semantic_preds).to(torch.long)
            unique_inds, counts = torch.unique(grid_pc, return_counts=True, dim=0)
            counts = counts.type(torch.long)

            grid_indices = [unique_inds[:, i] for i in range(grid_pc.shape[1])]
            update[grid_indices] = update[grid_indices] + counts
            self.relative_pose[:3,3] = self.relative_pose[:3,3] / self.voxel_sizes

            current_grid_inds = torch.floor((coords - self.min_bound) / self.voxel_sizes)
            current_grid_inds = torch.matmul(self.relative_pose[:3,:3], current_grid_inds[:, :3].T.to(torch.float)).T + self.relative_pose[:3,3] # relative coord
            current_grid_inds = current_grid_inds / self.grid_size
            # print(current_grid_inds)
            current_grid_inds = current_grid_inds.reshape(update.shape[0],update.shape[1],update.shape[2],-1).unsqueeze(0)
            # coords = torch.matmul(self.relative_pose[:3,:3], coords[:, :3].T.to(torch.float)).T + self.relative_pose[:3,3] # relative coord


            ## prior
            self.prior_map = torch.unsqueeze(self.prior_map.permute(3, 0, 1, 2), 0)
            grid_trilinear = F.grid_sample(self.prior_map, current_grid_inds,  align_corners=True, mode='bilinear')
            grid_trilinear = torch.squeeze(grid_trilinear,0).permute(1,2,3,0)
            print(torch.sum(grid_trilinear!=0))
            # update = update + grid_trilinear
            # 2: Apply BKI filters
            update = torch.unsqueeze(update.permute(3, 0, 1, 2), 0)
            update = self.ConvLayer(update)
            new_update = torch.squeeze(update).permute(1, 2, 3, 0)

            self.current_map = self.current_map + new_update  #+ grid_trilinear

            self.prior_map = self.current_map
            self.global_map = torch.cat((coords.reshape(-1, 3), self.current_map.reshape(-1,self.current_map.shape[-1])),dim=1)


        self.global_map = self.global_map.detach().cpu().numpy()
        return self.global_map



    # Propagate map given a transformation matrix
    def propagate(self, pose, prior_pose, Tr):
        pose = torch.from_numpy(pose).to(self.device)
        Tr = torch.from_numpy(Tr).to(self.device)
        prior_pose = torch.from_numpy(prior_pose).to(self.device)

        pose = torch.matmul(torch.linalg.inv(Tr), torch.matmul(pose, Tr))
        prior_pose = torch.matmul(torch.linalg.inv(Tr), torch.matmul(prior_pose, Tr))

        # Was just initialized
        if self.initial_pose is None:
            self.initial_pose = pose
            self.relative_pose = prior_pose
            return

        # GLobal frame transformation (to get from current frame to global frame)
        #global_pose = torch.matmul(torch.linalg.inv(pose), self.initial_pose)
        global_pose = pose
        self.relative_pose = torch.matmul(torch.linalg.inv(pose), torch.matmul(prior_pose, pose))
        translation = global_pose[:3, 3]
        # To select voxels from memory
        #translation_discretized = torch.round(translation / self.voxel_sizes) * self.voxel_sizes
        # Transformation on point cloud to account for discretization error
        #self.points_translation = translation - translation_discretized
        self.points_rotation = global_pose[:3, :3]

        self.translation_discretized = translation