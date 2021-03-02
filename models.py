"""
Author: John Sutor
Date: February 28, 2020

This file contains the base models for each object
detection architecture in this repository.
"""

import torch
import torch.nn as nn

from layers.yolov3 import ResBlock, ConvBlock


class YOLOv3(nn.Module):
    """
    The YOLOv3 architecture as proposed by Joseph Redmon and Ali Farhadi (2018).
    """

    def __init__(self, anchors: int, classes: int):
        """
        Arguments:
        boxes (int): number of boxes to subdivide the output into. Must
        be a divisor of the input shape (256 x 256)

        """
        super().__init__()

        self.anchors = anchors
        self.classes = classes

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.downsample = nn.ModuleList([
            # Input layer (512 x 512)
            ConvBlock(3, 32, 3, 1),
            ConvBlock(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            # First residual (128 x 128)
            ResBlock(32, 64),
            ConvBlock(64, 128, 3, 1),
            nn.MaxPool2d(2, 2),
            # Second residual (64 x 64)
            *[ResBlock(64, 128) for _ in range(2)],
            ConvBlock(128, 256, 3, 1),
            nn.MaxPool2d(2, 2),
            # Third residual (32 x 32)
            *[ResBlock(128, 256) for _ in range(8)],
            ConvBlock(256, 512, 3, 1),
            nn.MaxPool2d(2, 2),
            # Fourth residual (16 x 16)
            *[ResBlock(256, 512) for _ in range(8)],
            ConvBlock(512, 1024, 3, 1),
            nn.MaxPool2d(2, 2),
            # Fifth residual (8 x 8)
            *[ResBlock(512, 1024) for _ in range(4)],
        ])

        self.out_1 = nn.Sequential(
            ConvBlock(1024, 256, 3, 1),
            ConvBlock(256, anchors * (classes + 5), 1, 0) 
        )

        self.out_2 = nn.Sequential(
            ConvBlock(1536, 512, 3, 1),
            ConvBlock(512, anchors * (classes + 5), 1, 0)
        )

        self.out_3 = nn.Sequential(
            ConvBlock(1792, 512, 3, 1),
            ConvBlock(512, anchors * (classes + 5), 1, 0)
        )

    def forward(self, x):
        for l in self.downsample[:10]:
            x = l(x)
        
        conc_1 = x 

        for l in self.downsample[10:20]:
            x = l(x)

        conc_2 = x

        for l in self.downsample[20:]:
            x = l(x)

        out_1 = self.out_1(x)
        x = self.upsample(x)
        x = torch.cat((conc_2, x), dim=1)

        out_2 = self.out_2(x)
        x = self.upsample(x)
        x = torch.cat((conc_1, x), dim=1) 

        out_3 = self.out_3(x) 

        return (out_1, out_2, out_3)
