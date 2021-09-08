#-*-coding:utf-8-*-
"""
    Cost volume generation
    This is not going to be fast, do we have some faster way?
    Now I know why the decomposition model is so fast
    @author hqy
    @date 2021.9.8
"""

import torch
from torch import nn
from encoder import Encoder

class CostVolume(nn.Module):
    def __init__(self, res_num, max_disp = 200, use_cuda = True):
        super().__init__()
        self.encoder = Encoder(res_num)
        self.max_disp = max_disp
        self.cuda_flag = use_cuda
        self.conv = nn.Sequential(
            nn.Conv2d(max_disp, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 1, 3, 1, 1)
        )

    # input left(n, channel, w, h)
    # right should be padded on the left
    def costVolumeCalc(self, left:torch.Tensor, right:torch.Tensor) -> torch.Tensor:
        # n is usually 1, because the inputs don't have a unified size
        # in the dimension of width, padding
        # output padded is (c, h, w + max_disp), which should be reshaped to
        left = left.squeeze()
        right = right.squeeze()
        c, h, w = left.shape
        zeros = torch.zeros(c, h, self.max_disp)
        if self.cuda_flag:
            zeros = zeros.cuda()
        padded = torch.cat([zeros, right], dim = -1)            # output (c, h, w + max_disp)
        padded = padded.transpose(0, 2).unsqueeze(dim = -1)     # output (w + max_disp, h, c)
        left = left.transpose(0, 2).unsqueeze(dim = -2)         # output (w, h, 1, c)
        costs = []
        for i in range(self.max_disp):
            cost_map = left @ padded[i:i+w, :, :, :]          # output (w, h, 1, 1)
            costs.append(cost_map.squeeze(dim = -1))           # append (w, h, 1)
        costs = torch.cat(costs, dim = -1)                      # (w, h, max_disp)
        return costs.transpose(0, 2)                            # (max_disp, h, w)

    def forward(self, left, right):
        left = self.encoder(left)
        right = self.encoder(right)
        volume = self.costVolumeCalc(left, right)               # (b, max_disp, w, h)
        return self.conv(volume.unsqueeze(dim = 0)).squeeze(dim = 0)                                # (1, h, w)
