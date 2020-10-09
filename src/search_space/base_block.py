"""
@author: Guanghan Ning
@file: base_block.py
@time: 10/7/20 12:16 上午
@file_desc:
"""

import torch
import torch.nn as nn
from .base_network import Network

class Block(Network):
    """Base Block class."""

    def __init__(self):
        super().__init__()
        self.block = None

    def forward(self, x):
        return self.block(x)
