import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

R_WIDTH = 0.0592944
DEPTH_VAL = torch.from_numpy(np.arange(R_WIDTH, R_WIDTH * 128 + R_WIDTH, R_WIDTH)).float()

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, pooling = False):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

def huber_loss(depth_est, depth_gt):
    mask = depth_gt < R_WIDTH * 128
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
