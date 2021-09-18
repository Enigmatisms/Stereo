#-*-coding:utf-8-*-
"""
    Cost volume generation
    @author hqy (partly from the official implementation)
    @date 2021.9.13
"""

import torch
from torch import nn

def makeConv3d(in_chan, out_chan, k, stride, pad, use_norm = True, activate = True):
    blocks = [nn.Conv3d(in_chan, out_chan, k, stride, pad)]
    if use_norm:
        blocks.append(nn.BatchNorm3d(out_chan))
    if activate:
        blocks.append(nn.ReLU(True))
    return blocks

class CostVolume(nn.Module):
    def __init__(self, max_disp, in_chan):
        super().__init__()
        self.max_disp = max_disp
        # conv3d takes (_1, _2, _3, _4, _5) inputs, of which the second dimension is modified
        self.reg = nn.Sequential(
            *makeConv3d(in_chan, in_chan, 3, 1, 1),
            *makeConv3d(in_chan, in_chan, 3, 1, 1),
            *makeConv3d(in_chan, in_chan, 3, 1, 1),
            *makeConv3d(in_chan, in_chan, 3, 1, 1),

            *makeConv3d(in_chan, in_chan, 3, 1, 1),
            *makeConv3d(in_chan, in_chan, 3, 1, 1),
            *makeConv3d(in_chan, in_chan, 3, 1, 1),
            # output of course should be 1, do not let Conv3d scare you.
            *makeConv3d(in_chan, 1, 3, 1, 1, activate = False),
        )

    # input N C H W, transform to 5F N C D H W for Conv3d
    # output N D H W
    def forward(self, left:torch.Tensor, right:torch.Tensor):
        left_vol, right_vol = self.get_warped_feats_by_shift(left, right)
        vol = self.cost_computation_cor(left_vol, right_vol)       
        vol =  self.reg(vol).squeeze(dim = 1)       # N D H W
        prob_vol = torch.softmax(vol, dim = 1)      # softmax along disparity dimension
        disp_samples = self.generateDispSamples(prob_vol)
        return torch.sum(prob_vol * disp_samples, dim = 1)
    
    def generateDispSamples(self, volume:torch.Tensor):
        batch_num, _, h, w = volume.shape 
        single_layer = torch.arange(0, self.max_disp, dtype=volume.dtype, device=volume.device).view(-1, 1, 1).expand(-1, h, w)
        disp_samples = single_layer.repeat(batch_num, 1, 1, 1)
        return disp_samples

    # Credit goes to the official implementation
    # I merely understand how the following codes work, but didn't implement these
    def get_warped_feats_by_shift(self, left_feature_map:torch.Tensor, right_feature_map:torch.Tensor):
        """fucntion: build the warped feature volume. This version of the construction method takes up a little more memory.
        args:
            left_feature_map: feature maps from left view, N*C*H*W;
            right_feature_map: feature maps from right view, N*C*H*W;
            max_disp: maximum value of disparity, eg, 192;
        return:
            the warped feature volume, N*C*D*H*W;
        """
        batch_size, channels, height, width = right_feature_map.size()
        
        disp_samples = torch.arange(0.0, self.max_disp, device=right_feature_map.device)
        disp_samples = disp_samples.repeat(batch_size).view(batch_size, -1) # N*D
        
        x_coord = torch.arange(0.0, width, device=0).repeat(height).view(height, width)
        x_coord = torch.clamp(x_coord, min=0, max=width-1)
        x_coord = x_coord.expand(batch_size, -1, -1) # N*H*W
        
        right_x_coord = x_coord.expand(self.max_disp,-1,-1,-1).permute([1,0,2,3]) # N*D*H*W
        right_x_coord = right_x_coord - disp_samples.unsqueeze(-1).unsqueeze(-1)
        right_right_x_coord_tmp = right_x_coord.unsqueeze(1) # N*1*D*H*W
        right_x_coord = torch.clamp(right_x_coord, min=0, max=self.max_disp-1)
        
        left_vol = left_feature_map.expand(self.max_disp,-1,-1,-1,-1).permute([1,2,0,3,4])           # left feature map: direct copy
        right_vol = right_feature_map.expand(self.max_disp,-1,-1,-1,-1).permute([1,2,0,3,4])         # right featmap: copy and warp
        right_vol = torch.gather(right_vol, dim=4, index=right_x_coord.expand(channels,-1,-1,-1,-1).permute([1,0,2,3,4]).long()) # N*C*D*H*W
        right_vol = (1 - ((right_right_x_coord_tmp<0) + (right_right_x_coord_tmp>width-1)).float()) * (right_vol)     # over-boundary elems should be put to zero
        
        return left_vol, right_vol
    
    # Credit goes to the official implementation
    def cost_computation_cor(self, left_vol:torch.Tensor, right_vol:torch.Tensor) :
        """build the cost volume via correlation between left volume and right volume
        """
        cost_vol = left_vol.mul_(right_vol)
        return cost_vol         # output is N*C*D*H*W
