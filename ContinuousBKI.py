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

class_remap = np.array([
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    0, # None
    11,
    0,
    0,
    12,
    0,
    13,
    14,
    15,
    0,
    0,
    0,
    16,
    0,
    0,
    0,
    17,
    0,
    0,
    0,
    18,
    0,
    19,
    20
])

# Classes for RELLIS3d
colors = np.array([
    (0,0,0),        #void
    (108, 64, 20),  #dirt
    (0,102,0),      #grass
    (0,255,0),      #tree
    (0,153,153),    #pole
    (0,128,255),    #water
    (0,0,255),      #sky    
    (255,255,0),    #vehicle
    (255,0,127),    #object
    (64,64,64),     #asphalt
    (255,0,0),      #building
    (102,0,0),      #log
    (204,153,255),  #person
    (102, 0, 204),  #fence
    (255,153,204),  #bush
    (170,170,170),  #concrete
    (41,121,255),   #barrier
    (134,255,239),  #puddle
    (99,66,34),     #mud
    (110,22,138),   #rubble
    (255, 255, 255) # unknown
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

def publish_voxels(map, pub, centroids, min_dim, max_dim, grid_dims, model="DiscreteBKI"):
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
    
    semantic_map = torch.argmax(map, dim=-1).reshape(-1,)

    # Only publish nonfree voxels
    nonfree_mask = semantic_map!=0
    nonfree_centroids = centroids[nonfree_mask]
    nonfree_semantic_map = semantic_map[nonfree_mask]

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

    for i in range(nonfree_semantic_map.shape[0]):              
        pred = nonfree_semantic_map[i]

        point = Point32()
        color = ColorRGBA()
        point.x = nonfree_centroids[i, 0]
        point.y = nonfree_centroids[i, 1]
        point.z = nonfree_centroids[i, 2]

        color.r, color.g, color.b = colors[class_remap[pred]]

        color.a = 0.75
        marker.points.append(point)
        marker.colors.append(color)
    
    next_map.markers.append(marker)

    pub.publish(next_map)


def get_query_neighbors(queries, map):
    """
    Performs 3D convolution in a predefined cubic volume,
    returning K neighbors to each query point provided.

    Input:
        queries: Nx3 set of query points in new point cloud
        map: HxWxDx
    Output:
        
    """


