import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.cuda import amp

from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
import os

# TODO: reference for defining convolution based on spnn
# model = nn.Sequential(
#         spnn.Conv3d(4, 32, 3),
#         spnn.BatchNorm(32),
#         spnn.ReLU(True),
#         spnn.Conv3d(32, 64, 2, stride=2),
#         spnn.BatchNorm(64),
#         spnn.ReLU(True),
#         spnn.Conv3d(64, 64, 2, stride=2, transposed=True),
#         spnn.BatchNorm(64),
#         spnn.ReLU(True),
#         spnn.Conv3d(64, 32, 3),
#         spnn.BatchNorm(32),
#         spnn.ReLU(True),
#         spnn.Conv3d(32, 10, 1),
#     ).to(args.device)