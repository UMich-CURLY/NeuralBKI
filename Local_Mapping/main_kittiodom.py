import yaml
import os
from Data.KittiOdometry import *
from torch.utils.data import Dataset, DataLoader
import torch
from utils import *
import time
import rospy
from visualization_msgs.msg import MarkerArray
from tqdm import tqdm

DATA_CONFIG = "kitti_odometry"
MODEL_CONFIG = "ConvBKI_PerClass_Compound_02_odom"

# Data Parameters
data_params_file = os.path.join(os.getcwd(), "Configs", DATA_CONFIG + ".yaml")
with open(data_params_file, "r") as stream:
    try:
        data_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
# Model Parameters
model_params_file = os.path.join(os.getcwd(), "Configs", MODEL_CONFIG + ".yaml")
with open(model_params_file, "r") as stream:
    try:
        model_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

# CONSTANTS
SEED = model_params["seed"]
NUM_FRAMES = model_params["num_frames"]
# MODEL_RUN_DIR = os.path.join("Models", "Runs", MODEL_NAME + "_" + dataset)
NUM_WORKERS = model_params["num_workers"]
FLOAT_TYPE = torch.float32
LABEL_TYPE = torch.uint8
MAP_METHOD = model_params["map_method"]
LOAD_EPOCH = model_params["load_epoch"]
LOAD_DIR = model_params["save_dir"]
VISUALIZE = model_params["visualize"]
MEAS_RESULT = model_params["meas_result"]
GEN_PREDS = model_params["gen_preds"]
FROM_CONT = model_params["from_continuous"]
TO_CONT = model_params["to_continuous"]
PRED_PATH = model_params["pred_path"]


dataset = KittiOdomDataset(model_params["train"]["grid_params"], directory=data_params["data_dir"], device=dev,
                        num_frames=NUM_FRAMES, remap=False, use_aug=False, data_split=model_params["result_split"], from_continuous=FROM_CONT,
                        to_continuous=TO_CONT)

loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn, num_workers=10)


e2e_net = load_model(model_params, dev)

rospy.init_node('talker', anonymous=True)
map_pub = rospy.Publisher('SemMap_global', MarkerArray, queue_size=10)
next_map = MarkerArray()

# current_scene = dataset.scene_nums[0]
time_list = []
for idx in tqdm(range(len(dataset))):
    with torch.no_grad():
        # if dataset.scene_nums[idx] != current_scene:
        #     current_scene = dataset.scene_nums[idx]
        #     e2e_net.initialize_grid() 
        # Load data (Only one at a time)
        # in_dict = dataset[idx]
        # lidar = in_dict["lidar"][0]
        # seg_input, inv = generate_seg_in(lidar, model_params["res"])
        # lidar_pose = in_dict["poses"][0]
        # labels = in_dict["targets"][0]

        # Load data
        get_gt = model_params["result_split"] == "train" or model_params["result_split"] == "val" 
        pose, points, pred_labels, gt_labels, scene_id, frame_id = dataset.get_test_item(idx, get_gt=get_gt)

        input_data = [torch.tensor(pose).to(dev).type(dtype), torch.tensor(points).to(dev).type(dtype),
                      pred_labels]

        start_t = time.time()
        e2e_net(input_data)
        
        time_list.append(time.time() - start_t)

    if rospy.is_shutdown():
        exit("Closing Python")
    try:
        # reset_vis(map_pub)
        next_map = publish_local_map(e2e_net.grid, e2e_net.convbki_net.centroids, model_params["train"]["grid_params"]["voxel_sizes"],
                                     data_params["colors"], next_map, e2e_net.propagation_net.translation)
        map_pub.publish(next_map)
        rospy.sleep(0.5)
    except:
        exit("Publishing broke")

time_file = open("./time.txt", "w")
for t in time_list:
    time_file.write(str(t) + "\n")
time_file.close()