import roslib
import gc
import os
import sys
import pdb
import rospy
import numpy as np
import torch

# Adds parent dir to path
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from torch.utils.data import DataLoader
from Data.dataset import Rellis3dDataset, ray_trace_batch

from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float64MultiArray, String

sync_packet_received = None

def callback(data):
    global sync_packet_received 
    sync_packet_received = True
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

"""
Publish Format
0:  Frame index
1:  Number of points
2:  Number of classes per point
3a-3b: Flattened xyz coordinates for all points
3b-3c: Flattened softmax probabilities for all points
"""
def talker(base_fp, pkg_name, start_idx=450):
    pub = rospy.Publisher('preds', Float64MultiArray,queue_size=1)
    global sync_packet_received 
    rospy.Subscriber("sync", String, callback)

    rospy.init_node('talker', anonymous=True)
    # r = rospy.Rate(0.5) # This shoud be set so that it releases at a rate slower than the insertion of points, but making it as fast as possible will change the bottleneck of the program, tune it as mcuh as you want.  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rellis_ds = Rellis3dDataset(directory=base_fp, device=device, num_frames=10, remap=True, use_aug=False, model_setting="test")

    for idx in range(2769, len(rellis_ds)):
        points, points_labels, _, _, _ = rellis_ds[idx]
        scene_name  = rellis_ds._scenes_list[idx]
        scene_id    = int(scene_name)       # Scene ID
        frame_id    = rellis_ds._index_list[idx] # Frame ID in current scene ID

        pc_np = np.vstack(np.array(points))
        labels_np = np.vstack(np.array(points_labels))

        # Remove label 0
        void_mask = (labels_np!=0).reshape(-1)
        pc_np = pc_np[void_mask]
        labels_np = labels_np[void_mask]

        if pc_np.shape[0] <= 0:
            # Zero pad in case all labels are 0
            pc_np = np.zeros((1, 3))
            labels_np = np.ones((1, 1))
        fs_pc       = ray_trace_batch(pc_np, labels_np, 0.3, device)
        pc_np      = np.vstack( (pc_np, fs_pc[:, :3].reshape(-1, 3)))
        labels_np= np.vstack( (labels_np, fs_pc[:, 3].reshape(-1, 1)))
        pose = rellis_ds._poses[scene_id][frame_id]

        # print("pc_np shape ", pc_np.shape[0])
        data_info   = [idx, pc_np.shape[0], 21]
        pose_mat    = pose.reshape(3, 4) # rot | trans

        new_pc = (pose_mat[0:3, 0:3]@pc_np.T).T + pose_mat[:3, 3]
        # pdb.set_trace()
        if pkg_name == "semantic_bki":
            flat_coords = new_pc.reshape(-1).tolist()
            flat_preds  = labels_np.reshape(-1).tolist()
            flat_pose   = pose_mat.reshape(-1).tolist()

            final_data = data_info + flat_coords + flat_preds + flat_pose
            message = Float64MultiArray(data=final_data)

            pub.publish(message)
            # r.sleep()

        while not sync_packet_received:
            rospy.sleep(1)
        print("received sync packet")
        sync_packet_received = False
        gc.collect()


if __name__ == '__main__':
    # Change BASE_DIR to parent folder for Coords and Preds directories
    BASE_DIR = '/home/arthurzhang/Data/Rellis-3D'

    # main(coords_fp, preds_fp, poses_fp)
    talker(BASE_DIR, pkg_name="semantic_bki", start_idx=0)

    # with open('../catkin_ws/src/BKINeuralNet/data/carla_townheavy/labels/000000.label', 'rb') as f:
    #     label = np.fromfile(f, dtype=np.uint32)
    #     pdb.set_trace()
    # pdb.set_trace()
    # print(poses) 