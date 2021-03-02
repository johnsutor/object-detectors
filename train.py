"""
Author: John Sutor
Date: February 28, 2020

This file contains the training loop necessary for
training each respective object detection model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision

import models

model = models.YOLOv3(64, 80)
