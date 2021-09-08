#-*-coding:utf-8-*-
"""
    Stereo Matching encoder
    siamese network for feature extraction
    @author hqy
    @date 2021.9.8
"""
import torch
from torch import nn
from torch.nn import functional as F

def makeConv(in_chan, out_chan, k, stride, pad, use_norm = True):
    res = [nn.Conv2d(in_chan, out_chan, k, stride, pad)]
    if use_norm:
        res.append(nn.BatchNorm2d(out_chan))
    res.append(nn.ReLU(True))
    return res

class Encoder(nn.Module):
    def __init__(self, res_block_num = 3):
        super().__init__()
        self.conv_down = nn.Sequential(
            *makeConv(3, 64, 3, 1, 1),
        )
        # simple n block resnet 
        self.small_res = nn.ModuleList([])
        for i in range(res_block_num):
            self.small_res.append(nn.Sequential(*makeConv(64, 64, 3, 1, 1)))
            self.small_res.append(nn.Sequential(*makeConv(64, 64, 3, 1, 1, False)))
        self.conv_up = nn.Conv2d(64, 16, 3, 1, 1)
        self.res_block_num = 2 * res_block_num

    def forward(self, x):
        x = self.conv_down(x)
        for i in range(0, self.res_block_num, 2):
            conv1 = self.small_res[i]
            conv2 = self.small_res[i + 1]
            tmp = conv1(x)
            h = conv2(tmp)
            x = F.relu(h + x, True)
        return self.conv_up(x)
