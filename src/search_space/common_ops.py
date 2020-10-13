"""Frequently used operations and combos."""
import torch
import torch.nn as nn


class DilatedConv(nn.Module):
    """Class of Dilation Convolution."""

    def __init__(self, desc):
        super(DilatedConv, self).__init__()
        affine = desc.get('affine', True)
        relu = nn.ReLU(inplace=False)
        conv1 = nn.Conv2d(desc.channel_in, desc.channel_in, kernel_size=desc.kernel_size, stride=desc.stride,
                          padding=desc.padding, dilation=desc.dilation, groups=desc.channel_in, bias=False)
        conv2 = nn.Conv2d(desc.channel_in, desc.channel_out, kernel_size=1, padding=0, bias=False)
        bn = nn.BatchNorm2d(desc.channel_out, affine=affine)
        self.block = nn.Sequential(relu, conv1, conv2, bn)

    def forward(self, x):
        return self.block(x)


class SeparatedConv(nn.Module):
    """Class of Separated Convolution."""

    def __init__(self, desc):
        super(SeparatedConv, self).__init__()
        affine = desc.get('affine', True)
        relu1 = nn.ReLU(inplace=False)
        conv1 = nn.Conv2d(desc.channel_in, desc.channel_in, kernel_size=desc.kernel_size,
                          stride=desc.stride, padding=desc.padding, groups=desc.channel_in, bias=False)
        conv2 = nn.Conv2d(desc.channel_in, desc.channel_in, kernel_size=1, padding=0, bias=False)
        bn1 = nn.BatchNorm2d(desc.channel_in, affine=affine)
        relu2 = nn.ReLU(inplace=False)
        conv3 = nn.Conv2d(desc.channel_in, desc.channel_in, kernel_size=desc.kernel_size, stride=1,
                          padding=desc.padding, groups=desc.channel_in, bias=False)
        conv4 = nn.Conv2d(desc.channel_in, desc.channel_out, kernel_size=1, padding=0, bias=False)
        bn3 = nn.BatchNorm2d(desc.channel_out, affine=affine)
        self.block = nn.Sequential(relu1, conv1, conv2, bn1, relu2, conv3, conv4, bn3)

    def forward(self, x):
        return self.block(x)


class Identity(nn.Module):
    """Class of Identity operation."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    """Class of Zero operation."""

    def __init__(self, desc):
        super(Zero, self).__init__()
        self.stride = desc.stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    """Class of Factorized Reduce operation."""

    def __init__(self, desc):
        super(FactorizedReduce, self).__init__()
        if desc.channel_out % 2 != 0:
            raise Exception('channel_out must be divided by 2.')
        affine = desc.get('affine', True)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(desc.channel_in, desc.channel_out // 2, 1,
                               stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(desc.channel_in, desc.channel_out // 2, 1,
                               stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(desc.channel_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


def drop_path(x, prob):
    """

    Args:
        x (Tensor): input feature map
        prob (float): dropout probability

    Returns (Tensor): output feature map after dropout

    """
    if prob <= 0.:
        return x
    keep = 1. - prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep)
    x.div_(keep)
    x.mul_(mask)
    return x
