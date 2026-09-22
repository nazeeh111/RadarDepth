import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_parts import *

class CostRegNet(nn.Module):
    def __init__(self, max_channel = 8, custom_first_channel = False):
        super(CostRegNet, self).__init__()
        stride_num = 2
        padding = 1
        half_max_channel = int(max_channel / 2)
        quad_max_channel = int(max_channel / 4)

        first_kernel = 3
        first_padding = 1
        if custom_first_channel:
            first_kernel = (3, 5, 7)
            first_padding = (1, 2, 3)

        self.conv0 = ConvBnReLU3D(1, max_channel, kernel_size=first_kernel, pad=first_padding)

        self.conv1 = ConvBnReLU3D(max_channel, half_max_channel, stride=stride_num)
        self.conv2 = ConvBnReLU3D(half_max_channel, half_max_channel)

        self.conv3 = ConvBnReLU3D(half_max_channel, quad_max_channel, stride=stride_num)
        self.conv4 = ConvBnReLU3D(quad_max_channel, quad_max_channel)

        self.conv5 = ConvBnReLU3D(quad_max_channel, half_max_channel, stride=stride_num)
        self.conv6 = ConvBnReLU3D(half_max_channel, half_max_channel)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(half_max_channel, quad_max_channel, kernel_size=3, padding=1, output_padding=padding, stride=stride_num, bias=False),
            nn.BatchNorm3d(quad_max_channel),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(quad_max_channel, half_max_channel, kernel_size=3, padding=1, output_padding=padding, stride=stride_num, bias=False),
            nn.BatchNorm3d(half_max_channel),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(half_max_channel, max_channel, kernel_size=3, padding=1, output_padding=padding, stride=stride_num, bias=False),
            nn.BatchNorm3d(max_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(max_channel, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)

        return self.prob(x)

class SegmentationNet(nn.Module):
    def __init__(self):
        super(SegmentationNet, self).__init__()

        self.conv0 = ConvBnReLU3D(1, 8, (21, 3, 3), stride=(5, 1, 1), pad=(0, 1, 1))
        self.conv1 = ConvBnReLU3D(8, 8, (5, 3, 3), stride=(3, 1, 1), pad=(0, 1, 1))
        self.conv2 = ConvBnReLU3D(8, 8, (3, 3, 3), stride=(3, 1, 1), pad=(0, 1, 1))
        self.outc = nn.Conv3d(8, 1, 3, padding='same')

    def forward(self, x):
        return self.outc(self.conv2(self.conv1(self.conv0(x))))

def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth

class DRNet3D(nn.Module):
    def __init__(self, depth_values, segment = True) -> None:
        super(DRNet3D, self).__init__()

        self.cost_regularization = CostRegNet(max_channel=16, custom_first_channel=True)
        self.depth_values = depth_values

        self.segment = segment

        if segment:
            self.segmentation = SegmentationNet()

    def forward(self, radar_volume):
        cost_reg = self.cost_regularization(radar_volume)
        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=self.depth_values)

        if self.segment:
            seg = self.segmentation(prob_volume.unsqueeze(1)).squeeze(1)
        else:
            seg = None

        with torch.no_grad():
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(128, device=prob_volume.device, dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            std_vals = torch.std(prob_volume, dim=1)

        return {"depth" : depth, "prob_volume": photometric_confidence, "segment_mask" : seg, "std" : std_vals}
