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

# Intersection, union for one frame
def iou_one_frame(pred, target, n_classes=21):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection[cls] = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union[cls] = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection[cls]
    return intersection, union

def points_to_voxels_torch(voxel_grid, points, min_bound, grid_dims, voxel_sizes):
    voxels = torch.floor((points - min_bound) / voxel_sizes).to(dtype=torch.int)
    # Clamp to account for any floating point errors
    maxes = (grid_dims - 1).reshape(1, 3)
    mins = torch.zeros_like(maxes)
    voxels = torch.clip(voxels, mins, maxes).to(dtype=torch.long)

    voxel_grid = voxel_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    return voxel_grid


# Remap colors to np array 0 to 1
def remap_colors(colors):
    # color
    colors_temp = np.zeros((len(colors), 3))
    for i in range(len(colors)):
        colors_temp[i, :] = colors[i]
    colors = colors_temp.astype("int")
    colors = colors / 255.0
    return colors


def publish_voxels(map_object, min_dim, max_dim, grid_dims, colors, next_map):
    next_map.markers.clear()
    marker = Marker()
    marker.id = 0
    marker.ns = "Global_Semantic_Map"
    marker.header.frame_id = "map" # change this to match model + scene name LMSC_000001
    marker.type = marker.CUBE_LIST
    marker.action = marker.ADD
    marker.lifetime.secs = 0
    marker.header.stamp = rospy.Time.now()

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
    marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
    marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]

    semantic_labels = map_object.global_map[:,3:]

    semantic_labels = np.argmax(semantic_labels / np.sum(semantic_labels, axis=-1, keepdims=True), axis=-1)
    semantic_labels = semantic_labels.reshape(-1, 1)

    centroids = map_object.global_map[:,:3]
    for i in range(semantic_labels.shape[0]):
        pred = semantic_labels[i]
        point = Point32()
        color = ColorRGBA()
        point.x = centroids[i, 0]
        point.y = centroids[i, 1]
        point.z = centroids[i, 2]
        color.r, color.g, color.b = colors[pred].squeeze()

        color.a = 1.0
        marker.points.append(point)
        marker.colors.append(color)

    next_map.markers.append(marker)
    return next_map