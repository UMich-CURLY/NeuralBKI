import torch
from ConvBKI.ConvBKI import *
from Propagation.mapping_utils import *
from BKINet import *

from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA
import rospy
from visualization_msgs.msg import MarkerArray, Marker


def load_model(model_params, dev):
    
    prop_net = TransformWorldStatic(torch.tensor(model_params["train"]["grid_params"]["voxel_sizes"]).to(dev))

    grid_size = model_params["train"]["grid_params"]["grid_size"]
    min_bound = torch.tensor(model_params["train"]["grid_params"]["min_bound"]).to(dev)
    max_bound = torch.tensor(model_params["train"]["grid_params"]["max_bound"]).to(dev)
    num_classes = model_params["num_classes"]
    f = model_params["filter_size"]

    bki_layer = ConvBKI(grid_size, min_bound, max_bound,
                        filter_size=f, num_classes=num_classes, device=dev)

    e2e_net = BKINet(bki_layer, prop_net, grid_size, device=dev, num_classes=num_classes)
    return e2e_net


def publish_local_map(labeled_grid, centroids, voxel_dims, colors, next_map, translation, ignore_label=0):
    next_map.markers.clear()
    marker = Marker()
    marker.id = 0
    marker.ns = "Local Semantic Map"
    marker.header.frame_id = "map"
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.lifetime.secs = 0
    marker.header.stamp = rospy.Time.now()

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = voxel_dims[0]
    marker.scale.y = voxel_dims[1]
    marker.scale.z = voxel_dims[2]

    X, Y, Z, C = labeled_grid.shape
    semantic_labels = labeled_grid.view(-1, C).detach().cpu().numpy()
    centroids = centroids.detach().cpu().numpy()

    # Remove high variance points
    semantic_sums = np.sum(semantic_labels, axis=-1, keepdims=False)
    valid_mask = semantic_sums >= 0.1

    semantic_labels = semantic_labels[valid_mask, :]
    centroids = centroids[valid_mask, :]

    semantic_labels = np.argmax(semantic_labels / np.sum(semantic_labels, axis=-1, keepdims=True), axis=-1)
    semantic_labels = semantic_labels.reshape(-1, 1)

    # Filter out ignore label
    keep_mask = semantic_labels.squeeze() != ignore_label
    semantic_labels = semantic_labels[keep_mask, :]
    centroids = centroids[keep_mask, :]
    if centroids.shape[0] <= 1:
        return next_map

    # Offset to global coords
    centroids = centroids + translation.view(1, -1).detach().cpu().numpy()

    for i in range(semantic_labels.shape[0]):
        pred = semantic_labels.squeeze()[i]
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
    return next_map


def reset_vis(map_pub):
    marker_array_msg = MarkerArray()
    marker = Marker()
    marker.id = 0
    marker.ns = "Local Semantic Map"
    marker.action = Marker.DELETEALL
    marker_array_msg.markers.append(marker)
    map_pub.publish(marker_array_msg)