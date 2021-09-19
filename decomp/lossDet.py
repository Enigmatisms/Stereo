#-*-coding:utf-8-*-
"""
    Lost detail detection module for unsupervised learning
    So when is this network trained?
    @author Sentinel
    @date 2021.9.19
"""

import torch
from torch import nn
from torch.nn import functional as F

class LossDet(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_chan, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.convs(x)

# when to use this detail loss is unknown, maybe just during the upsampling process.
class DetailLossUnsupervised(nn.Module):
    def __init__(self, alpha:float):
        super().__init__()
        self.alpha = alpha

    """
        origin is the original feature map, upsample is the upsampled (feature loss) feature map
    """
    def forward(self, origin:torch.Tensor, upsample:torch.Tensor, mask:torch.Tensor):
        diff_maps = torch.norm(origin - upsample, dim = 1, keepdim = True)
        mask = F.relu((mask - 0.5) * 2.0)           # values < 0.5 are cut off
        fine_grain_num = torch.sum((mask > 0.0))
        if fine_grain_num == 0:
            raise RuntimeError("Ill posed question with mask values all below 0.5")
        s = torch.sum(diff_maps * mask) / fine_grain_num
        return fine_grain_num - self.alpha * s
        