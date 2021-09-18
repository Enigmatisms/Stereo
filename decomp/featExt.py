#-*-coding:utf-8-*-
"""
    Feature extraction network (partly resembles to Unet)
    @author Sentinel
    @date 2021.9.13
"""

import torch
from torch import nn

def makeConv2d(in_chan, out_chan, k, stride, pad, norm = True, activate = True):
    blocks = [nn.Conv2d(in_chan, out_chan, k, stride, pad)]
    if norm:
        blocks.append(nn.BatchNorm2d(out_chan))
    if activate:
        blocks.append(nn.ReLU(True))
    return blocks

"""
    Sizes of input KITTI images are set to be (1344 * 512) / 2
    Therefore they need 4 downsample
"""
class FeatExt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_down1 = nn.Sequential(            # output (n, 32, h / 2, w / 2)
            *makeConv2d(3, 32, 3, 1, 1, False),
            *makeConv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(2)
        )
        self.conv_down2 = nn.Sequential(             # output (n, 64, h / 4, w / 4)
            *makeConv2d(32, 64, 3, 1, 1),
            *makeConv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(2)
        )
        self.conv_down3 = nn.Sequential(             # output (n, 128, h / 8, w / 8)
            *makeConv2d(64, 128, 3, 1, 1),
            *makeConv2d(128, 128, 3, 1, 1),
            nn.MaxPool2d(2)
        )
        self.conv = nn.Sequential(                  # output (n, 128, h / 16, w / 16)
            *makeConv2d(128, 128, 3, 1, 1),
            nn.MaxPool2d(2)
        )
        self.conv_up1 = nn.Sequential(*makeConv2d(256, 64, 3, 1, 1))
        self.conv_up2 = nn.Sequential(*makeConv2d(128, 32, 3, 1, 1))
        self.conv_up3 = nn.Sequential(*makeConv2d(64, 32, 3, 1, 1))
        self.out0 = nn.Sequential(*makeConv2d(128, 32, 3, 1, 1, activate = False))
        self.out1 = nn.Sequential(*makeConv2d(64, 32, 3, 1, 1, activate = False))
        self.out2 = nn.Sequential(*makeConv2d(32, 32, 3, 1, 1, activate = False))
        self.out3 = nn.Sequential(*makeConv2d(32, 32, 3, 1, 1, activate = False))
        self.out4 = nn.Sequential(*makeConv2d(32, 32, 3, 1, 1, activate = False))
        self.upsample = nn.UpsamplingBilinear2d(scale_factor = 2.0)

    def forward(self, x0):
        feature_maps = []
        x1 = self.conv_down1(x0)            # output (n, 32, h / 2, w / 2)
        x2 = self.conv_down2(x1)            # output (n, 64, h / 4, w / 4)
        x3 = self.conv_down3(x2)            # output (n, 128, h / 8, w / 8)
        x4 = self.conv(x3)                  # output (n, 128, h / 16, w / 16)
        feature_maps.append(self.out0(x4))          # the smallest feature map
        x4 = self.upsample(x4)
        y1_cat = torch.cat([x3, x4], dim = 1)       # y1_cat is (n, 256, h/8, w/8)
        y1 = self.conv_up1(y1_cat)                  # y1 is (n, 64, h/8, w/8)
        feature_maps.append(self.out1(y1))
        y1 = self.upsample(y1)                      # y1 is (n, 256, h/4, w/4)
        y2_cat = torch.cat([x2, y1], dim = 1)       # y2_cat is (n, 128, h/4, w/4)
        y2 = self.conv_up2(y2_cat)                  # y2 is (n, 32, h/4, w/4)
        feature_maps.append(self.out2(y2))
        y2 = self.upsample(y2)                      # y2 is (n, 32, h/2, w/2)
        y3_cat = torch.cat([x1, y2], dim = 1)       # y3 is (n, 64, h/2, w/2)
        y3 = self.conv_up3(y3_cat)                  # y3 is (n, 32, h/2, w/2)
        feature_maps.append(self.out3(y3))      
        y4 = self.upsample(y3)                      # y4 is (n, 32, h, w)
        feature_maps.append(self.out4(y4))  # feature_maps is a bunch of 32-dim feature maps 
        return feature_maps
