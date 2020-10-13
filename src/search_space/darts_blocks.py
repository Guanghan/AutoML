import torch
import torch.nn as nn
from .base_block import Block
from .common_ops import Zero
from src.core.class_factory import NetworkType, ClassFactory
from src.search_space.base_network import Network
from src.search_space.common_ops import DilatedConv, SeparatedConv, Identity, FactorizedReduce


@ClassFactory.register(NetworkType.BLOCK)
class none(Block):
    """Class of none."""

    def __init__(self, desc):
        super(none, self).__init__()
        self.block = Zero(desc)


@ClassFactory.register(NetworkType.BLOCK)
class avg_pool_3x3(Block):
    """Class of 3x3 average pooling."""

    def __init__(self, desc):
        super(avg_pool_3x3, self).__init__()
        stride = desc.stride
        self.block = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)


@ClassFactory.register(NetworkType.BLOCK)
class max_pool_3x3(Block):
    """Class 3x3 max pooling."""

    def __init__(self, desc):
        super(max_pool_3x3, self).__init__()
        stride = desc.stride
        self.block = nn.MaxPool2d(3, stride=stride, padding=1)


@ClassFactory.register(NetworkType.BLOCK)
class skip_connect(Block):
    """Class of skip connect."""

    def __init__(self, desc):
        super(skip_connect, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        if desc.stride == 1:
            self.block = Identity()
        else:
            self.block = FactorizedReduce(desc)


@ClassFactory.register(NetworkType.BLOCK)
class sep_conv_3x3(Block):
    """Class of 3x3 separated convolution."""

    def __init__(self, desc):
        super(sep_conv_3x3, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 3
        desc.padding = 1
        self.block = SeparatedConv(desc)


@ClassFactory.register(NetworkType.BLOCK)
class sep_conv_5x5(Block):
    """Class of 5x5 separated convolution."""

    def __init__(self, desc):
        """Init sep_conv_5x5."""
        super(sep_conv_5x5, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 5
        desc.padding = 2
        self.block = SeparatedConv(desc)


@ClassFactory.register(NetworkType.BLOCK)
class sep_conv_7x7(Block):
    """Class of 7x7 separated convolution."""

    def __init__(self, desc):
        super(sep_conv_7x7, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 7
        desc.padding = 3
        self.block = SeparatedConv(desc)


@ClassFactory.register(NetworkType.BLOCK)
class dil_conv_3x3(Block):
    """Class of 3x3 dilation convolution."""

    def __init__(self, desc):
        super(dil_conv_3x3, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 3
        desc.padding = 2
        desc.dilation = 2
        self.block = DilatedConv(desc)


@ClassFactory.register(NetworkType.BLOCK)
class dil_conv_5x5(Block):
    """Class of 5x5 dilation convolution."""

    def __init__(self, desc):
        super(dil_conv_5x5, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 5
        desc.padding = 4
        desc.dilation = 2
        self.block = DilatedConv(desc)


@ClassFactory.register(NetworkType.BLOCK)
class conv_7x1_1x7(Block):
    """Class of 7x1 and 1x7 convolution."""

    def __init__(self, desc):
        super(conv_7x1_1x7, self).__init__()
        stride = desc.stride
        channel_in = desc.C
        channel_out = desc.C
        affine = desc.affine
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_out, (1, 7), stride=(1, stride),
                      padding=(0, 3), bias=False),
            nn.Conv2d(channel_in, channel_out, (7, 1), stride=(stride, 1),
                      padding=(3, 0), bias=False),
            nn.BatchNorm2d(channel_out, affine=affine)
        )


@ClassFactory.register(NetworkType.BLOCK)
class PreOneStem(Network):
    """Class of one stem convolution."""

    def __init__(self, desc):
        super(PreOneStem, self).__init__()
        self._C = desc.C
        self._stem_multi = desc.stem_multi
        self.C_curr = self._stem_multi * self._C
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.C_curr)
        )

    def forward(self, x):
        x = self.stem(x)
        return x, x

