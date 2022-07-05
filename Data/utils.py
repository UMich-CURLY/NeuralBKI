import os
import pdb
from matplotlib import markers
import rospy
import numpy as np
import time
import os
import pdb
import torch
from visualization_msgs.msg import *
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA


def points_to_voxels_torch(voxel_grid, points, min_bound, grid_dims, voxel_sizes):
    voxels = torch.floor((points - min_bound) / voxel_sizes).to(dtype=torch.int)
    # Clamp to account for any floating point errors
    maxes = (grid_dims - 1).reshape(1, 3)
    mins = torch.zeros_like(maxes)
    voxels = torch.clip(voxels, mins, maxes).to(dtype=torch.long)

    voxel_grid = voxel_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    return voxel_grid

def publish_voxels(map, pub, centroids, min_dim, 
    max_dim, grid_dims, model="DiscreteBKI", pub_dynamic=False,
    valid_voxels_mask=None):
    """
    Publishes voxel map over ros to be visualized in rviz
    Input:
        map: HxWxDxC voxel map where H=height, W=width,
            D=depth, C=num_classes 
        pub: rospy publisher handle
        centroids: H*W*Dx2 indices from centroids of voxel map
        min_dim: 3x1 minimum dimensions in xyz
        max_dim 3x1 maximum dimensions in xyz
        grid_dims: 3x1 voxel grid resolution in xyz
        model: name of model used (Default: DiscreteBKI)
    """
    if map.dim()==4:
        semantic_map = torch.argmax(map / torch.sum(map, dim=-1, keepdim=True), dim=-1, keepdim=True)
        semantic_map = semantic_map.reshape(-1, 1)
    else:
        semantic_map = map.reshape(-1, 1)

    if valid_voxels_mask!=None:
        valid_voxels_mask = valid_voxels_mask.reshape(-1)

    # Remove dynamic labels if specified
    if not pub_dynamic:
        dynamic_class = torch.tensor([
            0,
            7,
            8,
            12,
            20
        ], device=semantic_map.device).reshape(1, -1)
        # pdb.set_trace()
        dynamic_mask = torch.all(
            semantic_map.ne(dynamic_class), dim=-1
        )
        # pdb.set_trace()
        centroids = centroids[dynamic_mask]
        semantic_map = semantic_map[dynamic_mask]
  
        if valid_voxels_mask!=None:
            valid_voxels_mask = valid_voxels_mask[dynamic_mask]

    # Only publish nonfree voxels
    if valid_voxels_mask!=None:
        centroids = centroids[valid_voxels_mask]
        semantic_map = semantic_map[valid_voxels_mask].reshape(-1, 1)

    next_map = MarkerArray()

    marker = Marker()
    marker.id = 0
    marker.ns = model + "semantic_map"
    marker.header.frame_id = "map" # change this to match model + scene name LMSC_000001
    marker.type = marker.CUBE_LIST
    marker.action = marker.ADD
    
    # if frame == 0:
    #     marker.action = marker.ADD
    # else:
    #     marker.action = marker.MODIFY
    marker.lifetime.secs = 0
    # marker.header.stamp = 0

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
    marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
    marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]
    
    for i in range(semantic_map.shape[0]):              
        pred = semantic_map[i]

        point = Point32()
        color = ColorRGBA()
        point.x = centroids[i, 0]
        point.y = centroids[i, 1]
        point.z = centroids[i, 2]

        color.r, color.g, color.b = colors[pred]

        color.a = 1.0
        marker.points.append(point)
        marker.colors.append(color)
    
    next_map.markers.append(marker)

    pub.publish(next_map)


def publish_pc(pc, labels, pub, min_dim, 
    max_dim, grid_dims, model="DiscreteBKI", pub_dynamic=False, use_mask=True):
    """
    Publishes voxel map over ros to be visualized in rviz

    Input:
        map: HxWxDxC voxel map where H=height, W=width,
            D=depth, C=num_classes 
        pub: rospy publisher handle
        points: H*W*Dx2 indices from points of map
        min_dim: 3x1 minimum dimensions in xyz
        max_dim 3x1 maximum dimensions in xyz
        grid_dims: 3x1 voxel grid resolution in xyz
        model: name of model used (Default: DiscreteBKI)
    """
    if use_mask:
        # Only publish nonfree voxels
        nonfree_mask = (labels!=LABELS_REMAP[0]) & (labels!=LABELS_REMAP[-1])

        nonfree_points = pc[nonfree_mask]
        nonfree_labels = labels[nonfree_mask].reshape(-1, 1)

        # Remove dynamic labels if specified
        if not pub_dynamic:
            dynamic_labels = torch.from_numpy(DYNAMIC_LABELS).to(pc.device)
            dynamic_mask = torch.all(torch.ne(nonfree_labels, dynamic_labels), dim=-1)
            nonfree_points = nonfree_points[dynamic_mask]
            nonfree_labels = nonfree_labels[dynamic_mask].reshape(-1)
    else:
        nonfree_points = pc
        nonfree_labels=labels

    next_map = MarkerArray()

    marker = Marker()
    marker.id = 0
    marker.ns = model + "semantic_map"
    marker.header.frame_id = "map" # change this to match model + scene name LMSC_000001
    marker.type = marker.CUBE_LIST
    marker.action = marker.ADD
    marker.lifetime.secs = 0
    # marker.header.stamp = 0

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
    marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
    marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]
    
    for i in range(nonfree_labels.shape[0]):              
        pred = nonfree_labels[i].cpu().detach().numpy().astype(np.uint32)

        point = Point32()
        color = ColorRGBA()
        point.x = nonfree_points[i, 0]
        point.y = nonfree_points[i, 1]
        point.z = nonfree_points[i, 2]

        color.r, color.g, color.b = colors[pred]

        color.a = 1.0
        marker.points.append(point)
        marker.colors.append(color)
    
    next_map.markers.append(marker)
    pub.publish(next_map)



