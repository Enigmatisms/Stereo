#-*-coding:utf-8-*-
"""
    Lost detail detection module for unsupervised learning
    @author Sentinel
    @date 2021.9.13
"""

import torch
from torch import nn

class LossDet(nn.Module):
    def __init__(self):
        super().__init__()
        