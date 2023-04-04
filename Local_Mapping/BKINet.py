import torch

# BKINet consists of three components:
# 2) ConvBKI layer
# 3) Propagation method
# This module is intended for ROS integration


class BKINet(torch.nn.Module):
    def __init__(self, convbki_net, propagation_net, grid_size,
                 device="cpu", datatype=torch.float32, num_classes=20, prior=1e-6):
        super().__init__()
        self.convbki_net = convbki_net
        self.propagation_net = propagation_net

        self.grid_size = grid_size
        self.num_classes = num_classes
        self.dtype = datatype
        self.device = device
        self.prior = prior

        self.ego_to_map = torch.eye(4).to(device)
        self.grid = self.initialize_grid()

    def initialize_grid(self):
        self.propagation_net.reset()
        return torch.zeros(self.grid_size[0], self.grid_size[1], self.grid_size[2],
                           self.num_classes, device=self.device, requires_grad=True,
                           dtype=self.dtype) + self.prior

    def forward(self, input_data):
        '''
        Input:
        List of input for [propagation, segmentation]
        '''
        new_pose, lidar, point_labels = input_data

        # Propagate
        self.ego_to_map, self.grid = self.propagation_net(new_pose, self.grid)
        transformed_lidar = torch.matmul(self.ego_to_map[:3, :3], lidar[:, :3].T).T + self.ego_to_map[:3, 3]

        # Update
        point_labels = torch.tensor(point_labels).to(self.device)
        segmented_points = torch.concat((transformed_lidar[:, :3], point_labels), dim=1)
        self.grid = self.convbki_net(self.grid, segmented_points)