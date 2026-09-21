#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: model_module.py
Description: modules (SincNet, ConvNext, Depth-Wsie Separable, TCN) to be used by model_factory
Author: geoffroy.peeters@telecom-paris.fr
Edited by: yannik.venohr@uni-wuerzburg.de
"""
import torch
import torch.nn as nn

# ConvNeXt PAPER: https://arxiv.org/pdf/2201.03545
# ConvNeXt CODE: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, drop_path=0.0):
        super(ConvNeXtBlock, self).__init__()
        
        # 1. Depthwise convolution (spatial convolution with large kernel)
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels)
        
        # 2. Layer normalization applied across channels
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)  # LayerNorm is applied after permuting to (B, C, H, W)
        
        # 3. Pointwise convolution to project to higher dimensions (expanding and compressing channels)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)  # expand channels by 4x
        self.act = nn.GELU()  # GELU activation
        self.pwconv2 = nn.Linear(4 * in_channels, out_channels)  # project back to original channels
        
        # 4. Stochastic depth (optional) for better regularization
        self.drop_path = nn.Identity() if drop_path == 0 else StochasticDepth(drop_path)
    
    def forward(self, x):
        # Input: (B, C, H, W)
        residual = x

        # 1. Depthwise convolution
        x = self.dwconv(x)
        
        # 2. LayerNorm after permute to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        
        # 3. Pointwise convolutions + GELU
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 4. Drop path (if applicable) and residual connection
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = self.drop_path(x) + residual  # Add residual connection
        
        return x

class ConvNeXtBlockMPECL(nn.Module):
    """Channel-last variant of ConvNeXtBlockMPE.

    Input/Output: (B, C, F, T)
    Internally permutes to (B, T, F, C) so that nn.LayerNorm normalizes over
    (F, C) and nn.Linear operates on the channel dimension — matching the ConvNeXtBlock style. 
    """
    def __init__(self, in_channels, in_bins, out_channels, drop_path=0.0, k_t=7, k_f=7, **kwargs):
        super().__init__()


        # 1. Depthwise Conv (B, C, F, T)
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=(k_f, k_t),
                                padding=(k_f // 2, k_t // 2), groups=in_channels)

        # 2. Norm — applied in channel-last layout (B, T, F, C)
        self.norm = nn.LayerNorm([in_bins, in_channels], eps=1e-6, elementwise_affine=False)

        # 3. Pointwise ops via Linear with 2× expansion
        self.pwconv1 = nn.Linear(in_channels, 2 * in_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * in_channels, out_channels)

        self.drop_path = nn.Identity() if drop_path == 0 else StochasticDepth(drop_path)

    def forward(self, x):
        # x: (B, C, F, T)
        residual = x
        x = self.dwconv(x)

        x = x.permute(0, 3, 2, 1)   # (B, C, F, T) -> (B, T, F, C)
        x = self.norm(x)             # normalize 
        x = self.pwconv1(x)          # (B, T, F, C) -> (B, T, F, 4C)
        x = self.act(x)
        x = self.pwconv2(x)          # (B, T, F, 4C) -> (B, T, F, C_out)
        x = x.permute(0, 3, 2, 1)   # (B, T, F, C_out) -> (B, C_out, F, T)

        return self.drop_path(x) + residual


class StochasticDepth(nn.Module):
    """Drop paths (stochastic depth) per sample (when applied in the main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Sample binary mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask

