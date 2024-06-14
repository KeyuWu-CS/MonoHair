import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, ksize=3):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(ksize, ksize), padding=(ksize // 2, ksize // 2), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(ksize, ksize), padding=(ksize // 2, ksize // 2), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, ksize=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, ksize=ksize)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, ksize=3):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
            kernel_size=(ksize, ksize), stride=(2, 2), padding=(ksize // 2, ksize // 2), output_padding=(ksize // 2, ksize // 2))
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(ksize, ksize), padding=(ksize // 2, ksize // 2), bias=True)

    def forward(self, x):
        return self.conv(x)


class UNetSimple(nn.Module):
    def __init__(self, in_feat, ksize=3, num_chan=(16, 32, 64, 128)):
        super(UNetSimple, self).__init__()

        self.inc = DoubleConv(in_feat, num_chan[0], ksize)
        self.down1 = Down(num_chan[0], num_chan[1], ksize)
        self.down2 = Down(num_chan[1], num_chan[2], ksize)
        self.down3 = Down(num_chan[2], num_chan[3], ksize)
        self.up1 = Up(num_chan[3], num_chan[2], 3)
        self.up2 = Up(num_chan[2], num_chan[1], 3)
        self.up3 = Up(num_chan[1], num_chan[0], 3)
        self.output_feat = num_chan[0] + num_chan[1] + num_chan[2] + num_chan[3]


    def forward(self, x, masks, sample_coord):
        feat_init = self.inc(x)
        feat_d1 = self.down1(feat_init)
        feat_d2 = self.down2(feat_d1)
        feat_d3 = self.down3(feat_d2)
        feat_u1 = self.up1(feat_d3, feat_d2)
        feat_u2 = self.up2(feat_u1, feat_d1)
        feat_u3 = self.up3(feat_u2, feat_init)

        feats = [feat_d3, feat_u1, feat_u2, feat_u3]
        # masks_feats = F.grid_sample(masks, sample_coord, align_corners=False).squeeze(dim=3)
        sample_feats = torch.cat([F.grid_sample(feat, sample_coord, align_corners=False).squeeze(dim=3) for feat in feats], dim=1)
        # return sample_feats * masks_feats
        return sample_feats


    def get_feat(self, x):
        feat_init = self.inc(x)
        feat_d1 = self.down1(feat_init)
        feat_d2 = self.down2(feat_d1)
        feat_d3 = self.down3(feat_d2)
        feat_u1 = self.up1(feat_d3, feat_d2)
        feat_u2 = self.up2(feat_u1, feat_d1)
        feat_u3 = self.up3(feat_u2, feat_init)

        return [feat_d3, feat_u1, feat_u2, feat_u3]

class ShallowEncoder(nn.Module):
    def __init__(self, in_feat, ksize=3, num_chan=(16, 32, 64, 128)):
        super(ShallowEncoder, self).__init__()

        self.inc = DoubleConv(in_feat, num_chan[0], ksize)
        self.down1 = Down(num_chan[0], num_chan[1], ksize)
        self.down2 = Down(num_chan[1], num_chan[2], ksize)
        # self.down3 = Down(num_chan[2], num_chan[3], ksize)
        # self.up1 = Up(num_chan[3], num_chan[2], 3)
        # self.up2 = Up(num_chan[2], num_chan[1], 3)
        # self.up3 = Up(num_chan[1], num_chan[0], 3)
        self.output_feat = num_chan[0] + num_chan[1] + num_chan[2]


    def forward(self, x, masks, sample_coord):
        feat_init = self.inc(x)
        feat_d1 = self.down1(feat_init)
        feat_d2 = self.down2(feat_d1)
        # feat_d3 = self.down3(feat_d2)
        # feat_u1 = self.up1(feat_d3, feat_d2)
        # feat_u2 = self.up2(feat_u1, feat_d1)
        # feat_u3 = self.up3(feat_u2, feat_init)

        feats = [feat_init, feat_d1, feat_d2]
        # masks_feats = F.grid_sample(masks, sample_coord, align_corners=False).squeeze(dim=3)
        sample_feats = torch.cat([F.grid_sample(feat, sample_coord, align_corners=False).squeeze(dim=3) for feat in feats], dim=1)
        # return sample_feats * masks_feats
        return sample_feats


    def get_feat(self, x):
        feat_init = self.inc(x)
        feat_d1 = self.down1(feat_init)
        feat_d2 = self.down2(feat_d1)
        # feat_d3 = self.down3(feat_d2)
        # feat_u1 = self.up1(feat_d3, feat_d2)
        # feat_u2 = self.up2(feat_u1, feat_d1)
        # feat_u3 = self.up3(feat_u2, feat_init)

        return [feat_init, feat_d1, feat_d2]
