import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ConvBlock

class DownModule(nn.Module):

    def __init__(self, in_feat, out_feat, kernel=3, dropprob=0.0, repeat=1):
        super(DownModule, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = ConvBlock(in_feat, out_feat, kernel=kernel, padding=kernel // 2, dropprob=dropprob, use_res=True if in_feat == out_feat else False)
        self.convs = nn.ModuleList([ConvBlock(out_feat, out_feat, kernel=kernel, padding=kernel // 2, dropprob=dropprob) for _ in range(repeat)])

    def forward(self, x, sample_coord):
        '''

        :param x: [V, C_in, H, W], input tensor
        :param sample_coord: [V, N, 1, 2], N = num of sample points
        :param mask_balance: [V, 1, H, W]
        :return: [V, C_in, N]
        '''
        y = self.pool(x)
        y = self.conv1(y)
        for layer in self.convs:
            y = layer(y)
        sample_feat = F.grid_sample(y, sample_coord, align_corners=False).squeeze(dim=3)

        return sample_feat, y

    def get_feat(self, x):
        y = self.pool(x)
        y = self.conv1(y)
        for layer in self.convs:
            y = layer(y)

        return y


def init_weight(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class Shallow(nn.Module):

    def __init__(self, in_feat, kernel=3):
        super(Shallow, self).__init__()
        self.init_pool = nn.MaxPool2d(2, stride=2)
        # self.avg_pool2 = nn.AvgPool2d(2, stride=2)
        self.init_conv = ConvBlock(in_feat, 32, kernel=kernel, padding=kernel // 2, dropprob=0.0, use_res=False)

        self.layer1 = DownModule(32, 32, kernel=kernel)
        self.layer2 = DownModule(32, 64, kernel=kernel)
        # self.layer3 = DownModule(32, 64, kernel=kernel, repeat=2)
        # self.layer4 = DownModule(128, 128, kernel=kernel, repeat=2)
        # self.layer5 = DownModule(128, 128, kernel=kernel, repeat=4)
        # self.layer6 = DownModule(128, 128)
        # self.layer7 = DownModule(128, 128)

        self.apply(init_weight)
        self.output_feat = 32 + 32 + 64

    def getBalanceMask(self, mask, kernel_size=3):
        '''
        no positive effects, currently unused
        :param mask: [N, 1, H, W], binary values, 1.0 (active) or 0.0
        :return: [N, 1, H, W]
        '''
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask)
        weights = F.conv2d(mask, kernel, padding=kernel_size // 2)
        balance = torch.zeros(weights.shape).to(mask)
        balance[weights > 0] = weights.max() / weights[weights > 0]

        return balance

    def forward(self, x, masks, sample_coord):
        # [V, 1, N]
        masks_feat = F.grid_sample(masks, sample_coord, align_corners=False).squeeze(dim=3)

        y = self.init_pool(x)
        y = self.init_conv(y)
        sample_feat0 = F.grid_sample(y, sample_coord, align_corners=False).squeeze(dim=3)

        sample_feat1, y = self.layer1(y, sample_coord)

        sample_feat2, y = self.layer2(y, sample_coord)

        # sample_feat3, y = self.layer3(y, sample_coord)

        # sample_feat4, y = self.layer4(y, sample_coord)
        #
        # sample_feat5, y = self.layer5(y, sample_coord)

        sample_feats = torch.cat([sample_feat0, sample_feat1, sample_feat2], dim=1)
        return sample_feats * masks_feat

    def get_feat(self, x):
        y = self.init_pool(x)
        feat0 = self.init_conv(y)
        feat1 = self.layer1.get_feat(feat0)
        feat2 = self.layer2.get_feat(feat1)

        return [feat0, feat1, feat2]


