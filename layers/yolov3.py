"""
Author: John Sutor
Date: February 28, 2020

This file contains utility base layers for use within the
YOLOv3 architecture.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Basic YOLOv3 residual block
    """

    def __init__(self, low: int, high: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(high, low, 1, bias=False),
            nn.BatchNorm2d(low),
            nn.LeakyReLU(0.1),
            nn.Conv2d(low, high, 3, 1, 1, bias=False),
            nn.BatchNorm2d(high),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        res = x
        x = self.layer(x)

        return x + res

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.layer(x)


