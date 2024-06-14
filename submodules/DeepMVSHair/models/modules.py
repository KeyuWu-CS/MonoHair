import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):

    def __init__(self, channel, reduce=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduce, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduce, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # batch and channel nums
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):

    def __init__(self, input_channel, output_channel, kernel=3, stride=1, padding=1, norm_type='in',
                 dilation=1, bias=True, dropprob=0.05, use_se=True, use_res=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, bias=bias)
        if norm_type == 'in':
            self.norm = nn.InstanceNorm2d(output_channel)
        elif norm_type == 'bn':
            self.norm = nn.BatchNorm2d(output_channel)
        else:
            self.norm = None

        self.se = SqueezeExcite(output_channel) if use_se else None
        self.dropout = nn.Dropout(p=dropprob)
        self.use_res = use_res

    def forward(self, x):
        y = self.conv(x)

        if self.norm is not None:
            y = self.norm(y)

        if self.se is not None:
            y = self.se(y) # squeeze and excitation

        if self.dropout.p != 0:
            y = self.dropout(y)

        if self.use_res:
            y = y + x # residual

        return F.relu(y)


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, use_se = False, norm_type='in'):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        if norm_type == 'in':
            self.norm = nn.InstanceNorm2d(noutput)
        elif norm_type == 'bn':
            self.norm = nn.BatchNorm2d(noutput)
        else:
            self.norm = None
        self.se = SqueezeExcite(noutput) if use_se else None

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)

        if self.norm is not None:
            output = self.norm(output)
        if self.se is not None:
            output = self.se(output)
        return F.relu(output)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, use_se = False, norm_type='in'):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)

        if norm_type == 'in':
            self.norm = nn.InstanceNorm2d(noutput)
        elif norm_type == 'bn':
            self.norm = nn.BatchNorm2d(noutput)
        else:
            self.norm = None

        self.se = SqueezeExcite(noutput) if use_se else None

    def forward(self, input):
        output = self.conv(input)

        if self.norm is not None:
            output = self.norm(output)
        if self.se is not None:
            output = self.se(output)
        return F.relu(output)