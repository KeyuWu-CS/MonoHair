import os
import sys
#
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ConvBlock, DownsamplerBlock, UpsamplerBlock


class UnetEncoder(nn.Module):

    def __init__(self, input_channels=2, kernel_size=5, depth_channels=(16, 64, 128, 256), use_se=True):
        super(UnetEncoder, self).__init__()
        self.init_conv = ConvBlock(input_channels, depth_channels[0], kernel=kernel_size, padding=kernel_size // 2, dropprob=0, use_res=False)

        # self.layer1 = nn.ModuleList()
        # self.layer1.append(ConvBlock(depth_channels[0], depth_channels[0]))
        self.down1 = DownsamplerBlock(depth_channels[0], depth_channels[1], use_se=use_se)

        self.layer2 = nn.ModuleList()
        self.layer2.append(ConvBlock(depth_channels[1], depth_channels[1], kernel=kernel_size, padding=kernel_size // 2))
        self.down2 = DownsamplerBlock(depth_channels[1], depth_channels[2], use_se=use_se)

        self.layer3 = nn.ModuleList()
        self.layer3.append(ConvBlock(depth_channels[2], depth_channels[2], kernel=kernel_size, padding=kernel_size // 2, dropprob=0.1))
        self.layer3.append(ConvBlock(depth_channels[2], depth_channels[2], kernel=kernel_size, padding=kernel_size // 2, dropprob=0.1))
        self.down3 = DownsamplerBlock(depth_channels[2], depth_channels[3], use_se=use_se)

        self.layer4 = nn.ModuleList()
        self.layer4.append(ConvBlock(depth_channels[3], depth_channels[3], kernel=kernel_size, padding=kernel_size // 2, dropprob=0.1))
        self.layer4.append(ConvBlock(depth_channels[3], depth_channels[3], kernel=kernel_size, padding=kernel_size // 2, dropprob=0.1))
        # self.layer4.append(ConvBlock(depth_channels[3], depth_channels[3], dropprob=0.3))
        # self.layer4.append(ConvBlock(depth_channels[3], depth_channels[3], dropprob=0.3))


    def forward(self, x):

        y = self.init_conv(x)
        # for layer in self.layer1:
        #     y = layer(y)
        sup1 = y

        y = self.down1(y)
        for layer in self.layer2:
            y = layer(y)
        sup2 = y

        y = self.down2(y)
        for layer in self.layer3:
            y = layer(y)
        sup3 = y

        y = self.down3(y)
        for layer in self.layer4:
            y = layer(y)

        return y, [sup3, sup2, sup1]


class UnetDecoder(nn.Module):

    def __init__(self, kernel_size=5, depth_channels=(256, 128, 64, 16), sup_channels=(128, 64, 16), use_se=True):
        super(UnetDecoder, self).__init__()
        self.up1 = UpsamplerBlock(depth_channels[0], depth_channels[1], use_se=use_se)
        self.layer1 = nn.ModuleList()
        self.layer1.append(ConvBlock(depth_channels[1] + sup_channels[0], depth_channels[1],
                                     kernel=kernel_size, padding=kernel_size // 2, dropprob=0, use_res=False))
        self.layer1.append(ConvBlock(depth_channels[1], depth_channels[1],
                                     kernel=kernel_size, padding=kernel_size // 2, dropprob=0))
        # self.layer1.append(ConvBlock(depth_channels[1], depth_channels[1], dropprob=0))

        self.up2 = UpsamplerBlock(depth_channels[1], depth_channels[2], use_se=use_se)
        self.layer2 = nn.ModuleList()
        self.layer2.append(ConvBlock(depth_channels[2] + sup_channels[1], depth_channels[2],
                                     kernel=kernel_size, padding=kernel_size // 2, dropprob=0, use_res=False))
        self.layer2.append(ConvBlock(depth_channels[2], depth_channels[2],
                                     kernel=kernel_size, padding=kernel_size // 2, dropprob=0))
        # self.layer2.append(ConvBlock(depth_channels[2], depth_channels[2], dropprob=0))

        self.up3 = UpsamplerBlock(depth_channels[2], depth_channels[3], use_se=use_se)
        self.layer3 = nn.ModuleList()
        self.layer3.append(ConvBlock(depth_channels[3] + sup_channels[2], depth_channels[3],
                                     kernel=kernel_size, padding=kernel_size // 2, dropprob=0, use_res=False))
        # self.layer3.append(ConvBlock(depth_channels[3], depth_channels[3], dropprob=0))

    def forward(self, x, sup_list):

        y = self.up1(x)
        y = torch.cat([sup_list[0], y], dim=1)
        for layer in self.layer1:
            y = layer(y)

        y = self.up2(y)
        y = torch.cat([sup_list[1], y], dim=1)
        for layer in self.layer2:
            y = layer(y)

        y = self.up3(y)
        y = torch.cat([sup_list[2], y], dim=1)
        for layer in self.layer3:
            y = layer(y)

        return y


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class Unet(nn.Module):

    def __init__(self, input_channels=2, output_channels=2, kernel_size=3, depth_channels=(16, 32, 64, 128)):
        super(Unet, self).__init__()
        self.encoder = UnetEncoder(input_channels=input_channels, kernel_size=kernel_size, depth_channels=depth_channels)
        self.decoder = UnetDecoder(kernel_size=kernel_size, depth_channels=depth_channels[::-1], sup_channels=depth_channels[:-1][::-1])
        self.output_conv = nn.Conv2d(in_channels=depth_channels[0], out_channels=output_channels, kernel_size=kernel_size,
                                     padding=kernel_size // 2)

        self.apply(init_weights)

    def forward(self, x):
        y, sups = self.encoder(x)
        output = self.output_conv(self.decoder(y, sups))
        return output


if __name__ == '__main__':
    sample = torch.rand(3, 2, 720, 960).cuda()
    net = Unet().cuda()
    out = net(sample)
    print('out', sample.shape)