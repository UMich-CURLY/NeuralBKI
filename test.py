#!/usr/bin/env python
from matplotlib import markers
import rospy
import numpy as np
import time
import os
import json
import pdb
from visualization_msgs.msg import *
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import cv2
import gc
def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1
    return uncompressed
LABELS_REMAP = np.array([
    0,  #void
    1,  #dirt
    0,  #none
    2,  #grass
    3,  #tree
    4,  #pole
    5,  #water
    6,  #sky
    7,  #vehicle
    8,  #object
    9,  #asphalt
    0,  # None
    10, #building
    0,  
    0,
    11, #log
    0,
    12, #person
    13, #fence
    14, #bush
    0,
    0,
    0,
    15, #concrete
    0,
    0,
    0,
    16, #barrier
    0,
    0,
    0,
    17, #puddle
    0,
    18, #mud
    19,  #rubble
    0,
    0,
    0,
    0,
    0,
    20
    
])
colors = np.array([ # RGB
    (0,0,0),        #0void
    (108, 64, 20),  #1dirt
    (0,102,0),      #2grass
    (0,255,0),      #3tree
    (0,153,153),    #4pole
    (0,128,255),    #5water
    (0,0,255),      #6sky    
    (255,255,0),    #7vehicle
    (255,0,127),    #8object
    (64,64,64),     #9asphalt
    (255,0,0),      #10building
    (102,0,0),      #11log
    (204,153,255),  #12person
    (102, 0, 204),  #13fence
    (255,153,204),  #14bush
    (170,170,170),  #15concrete
    (41,121,255),   #16barrier
    (134,255,239),  #17puddle
    (99,66,34),     #18mud
    (110,22,138),    #19rubble
    (255,255,0)     #20freespace
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses
def point_s(self):
    point_s.x = self[0]
    point_s.y = self[1]
    point_s.z = self[2]
    return point_s
def color_s(self):
    color_s.r = self[0]
    color_s.g = self[1]
    color_s.b = self[2]
    color_s.a = self[2]
    return color_s
def talker():
    model_names = ['MotionSC', 'LMSC', 'SSC_Full', 'JS3C']
    model_offsets = [
        # [-35, 35.0, 0.0],
        # [-35, -35.0, 0.0],
        # [35, 35.0, 0.0 ],
        # [35.0, -35.0, 0.0]
        # [-30, 29.0, 0.0],
        # [-30, -28.5, 0.0],
        # [30, 29.0, 0.0 ],
        # [30.0, -28.5, 0.0]
        [0, 0, 0]
    ]
    pub_MotionSC = rospy.Publisher('MotionSC_mapper', MarkerArray, queue_size=10)
    pub = [pub_MotionSC]
    rospy.init_node('talker',disable_signals=True)
    # while not rospy.is_shutdown():
    MotionSC_markers = MarkerArray()
    markers = [MotionSC_markers]
    data_dir = '/home/arthurzhang/Data/Rellis-3D/00000/voxels'
    load_dir_MotionSC = data_dir
    load_dirs = [load_dir_MotionSC]
    print("load_dirs:", load_dirs)
    
    grid_size = [256.0, 256.0, 16.0]
    eval_size = list(np.uint32(grid_size))
    
    # points
    min_bound = [-25.6, -25.6, -2.0]
    max_bound = [25.6, 25.6,  1.2]
    coor_ranges = min_bound + max_bound
    voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / grid_size[0], 
                  abs(coor_ranges[4] - coor_ranges[1]) / grid_size[1],
                  abs(coor_ranges[5] - coor_ranges[2]) / grid_size[2]]
    x = np.linspace(min_bound[0], max_bound[0], num=int(grid_size[0])) + voxel_sizes[0] / 2
    y = np.linspace(min_bound[1], max_bound[1], num=int(grid_size[1])) + voxel_sizes[1] / 2
    z = np.linspace(min_bound[2], max_bound[2], num=int(grid_size[2])) + voxel_sizes[2] / 2
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    
    xv = xv.reshape(-1,) # 128x128x8 -> N
    yv = yv.reshape(-1,)
    zv = zv.reshape(-1,)
    
    # model_labels = [None] * len(load_dirs) # index i has labels for model i
    frames = sorted(os.listdir(load_dir_MotionSC))
    num_files = int(len(frames) / 4)
    #for frame in range(int(num_files)):
    for frame in range(1):
        # print("Elapsed time since last frame: ", time.time() - start_time)
        print(frame)
        #MotionSC_markers.clear() 
        MotionSC_markers.markers.clear()
        
        for model in range (len(load_dirs)):
            points = np.stack((xv, yv, zv), axis=1) # Nx3 
            frame_filepath = os.path.join(load_dirs[model], str(frame).zfill(6) + '.label')
            label = np.fromfile(frame_filepath, dtype=np.uint8)
            # label = unpack(np.fromfile(frame_filepath, dtype=np.uint8))
            # label = np.fromfile(frame_filepath, dtype=np.uint16).astype(np.uint32)
            
            model_labels = LABELS_REMAP[label]
            
            non_free = model_labels != 0 # 128x128x8
            points = points[non_free, :]
            model_labels= model_labels[non_free]
            # non_free2 = model_labels == 15
            # points = points[non_free2, :]
            # model_labels= model_labels[non_free2]
            # model_labels[model] = np.fromfile(frame_filepath + ".label", dtype="uint32").reshape(grid_size) # 128x128x8
            # model_labels[model] = model_labels[model].reshape(-1,)
            # swap axes
            new_points = np.zeros(points.shape)
            new_points[:, 0] = points[:, 1]
            new_points[:, 1] = points[:, 0]
            new_points[:, 2] = points[:, 2]
            points = new_points
            ### Cube list
            marker = Marker()
            marker.id = model
            marker.ns = model_names[model] + "basic_shapes"
            marker.header.frame_id = "map"# change this to match model + scene name LMSC_000001
            marker.type = marker.CUBE_LIST
            marker.action = marker.ADD
            marker.lifetime.secs = 0
            marker.header.stamp = rospy.Time.now()
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1
            # Scale (meters)
            marker.scale.x = (max_bound[0] - min_bound[0]) / grid_size[0]
            marker.scale.y = (max_bound[1] - min_bound[1]) / grid_size[1]
            marker.scale.z = (max_bound[2] - min_bound[2]) / grid_size[2]
            points[:,0] += model_offsets[model][0]
            points[:,1] += model_offsets[model][1]
            points[:,2] += model_offsets[model][2]
            values, counts = np.unique(model_labels, return_counts=True)
            print(values, counts)
            for i in range(model_labels.shape[0]):                   
                pred = model_labels[i]
                point = Point32()
                color = ColorRGBA()
                point.x = points[i,0]
                point.y = points[i,1]
                point.z = points[i,2]
                color.r, color.g, color.b = colors[pred]
                
                color.a = 1.0
                if pred == 20:
                    color.a = 0
                
                marker.points.append(point)
                marker.colors.append(color)
            markers[model].markers.append(marker)

        for model in range(len(load_dirs)):
            pub[model].publish(markers[model])
        gc.collect()
if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:      
        pass    