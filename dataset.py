import argparse
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.cuda import amp

from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.collate import sparse_collate

from torchsparse.utils.quantize import sparse_quantize
import os

class KittiDataset:

    def __init__(self, input_size: int, voxel_size: float, data_dir='/home/neofelis/jingyu/sim_to_real/dataset', 
        max_extent=[25.6,25.6,4], min_extent=[-25.6,-25.6,-2], seq='04', device='cuda') -> None:
        self.input_size = input_size
        self.voxel_size = voxel_size
        self.data_dir = data_dir
        self.max_extent = max_extent
        self.min_extent = min_extent
        self.seq_num = seq
        self.device = device
        self.LABEL_MAP = np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 0, 1, 19,
                        19, 19, 2, 19, 19, 3, 19, 4, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 5, 6, 7, 19, 19, 19, 19, 19, 19,
                        19, 8, 19, 19, 19, 9, 19, 19, 19, 10, 11, 12, 13,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 14, 15, 16, 19, 19, 19, 19, 19,
                        19, 19, 17, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19])

        self.lidar_folder = os.path.join(self.data_dir, 'sequences', self.seq_num, 'velodyne')
        self.label_folder = os.path.join(self.data_dir, 'sequences', self.seq_num, 'labels')
        self.lidar_files = sorted(os.listdir(self.lidar_folder))
        self.label_files = sorted(os.listdir(self.label_folder))


    def __getitem__(self, i: int) -> Dict[str, Any]:
        # Used to get the lidar and label for running the SPVNAS
        lidar = np.fromfile(os.path.join(self.lidar_folder, self.lidar_files[i]), dtype=np.float32)
        label = np.fromfile(os.path.join(self.label_folder, self.label_files[i]), dtype=np.int32)
        label = self.process_label(label)

        # Filter out ignored points
        lidar = lidar[label != 19]
        label = label[label != 19]
        
        inputs = np.random.uniform(-25.6, 25.6, size=(self.input_size, 4))
        labels = np.random.choice(10, size=self.input_size)

        # Quantize coordinates
        coords = np.round(lidar[:, :3] / 0.05)
        coords -= coords.min(0, keepdims=1)
        feats = lidar

        coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)

        inputs = SparseTensor(coords=coords, feats=feats)
        inputs = sparse_collate([inputs]).to(self.device)

        return inputs

        coords, feats = inputs[:, :3], inputs
        coords -= np.min(coords, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords,
                                          self.voxel_size,
                                          return_index=True)

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)
        labels = torch.tensor(labels[indices], dtype=torch.long)

        input = SparseTensor(coords=coords, feats=feats)
        label = SparseTensor(coords=coords, feats=labels)
        return {'input': input, 'label': label}

    def process_label(self, label):
        label = label & 0xFFFF
        label = self.LABEL_MAP[label & 0xFFFF]
        return label

    def __len__(self):
        return len(self.lidar_files)

