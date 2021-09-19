#-*-coding:utf-8-*-
"""
    Sparse matching module
    The purpose of this module is to calculate the probability volume
    Using the proba volume to obtain the disparity expectation
    The official implementation is done in CUDA (.cu)
    @author Sentinel
    @date 2021.9.19
"""

import torch
from torch import nn

"""
    This should be a parallel program, yet direct python implementation inhibits concurrency
    Without CUDA, the pixel-wise operation will be slow.
"""
class SparseMatch(nn.Module):
    def __init__(self):
        super().__init__()
