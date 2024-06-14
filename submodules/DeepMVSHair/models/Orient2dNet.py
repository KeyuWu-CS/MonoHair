import os
import sys
#
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Unet import Unet

class Orient2dNet(nn.Module):

    def __init__(self, input_channels=2, output_channels=2, kernel_size=3, depth_channels=(16, 32, 64, 128), with_gt=True):
        super(Orient2dNet, self).__init__()
        self.Unet = Unet(input_channels=input_channels, output_channels=output_channels, kernel_size=kernel_size, depth_channels=depth_channels)
        self.with_gt = with_gt

    def forward(self, data):
        if self.with_gt:
            return self.forward_with_gt(data['input'], data['target'], data['mask'])
        else:
            return self.forward_raw(data['input'])

    def forward_with_gt(self, input, target, mask):
        raw_output = self.Unet(input)
        output = F.normalize(raw_output, dim=1)
        mask = mask.expand_as(output)
        loss = F.l1_loss(output[mask], target[mask])

        return loss, output

    def forward_raw(self, input):
        raw_output = self.Unet(input)
        output = F.normalize(raw_output, dim=1)
        return output