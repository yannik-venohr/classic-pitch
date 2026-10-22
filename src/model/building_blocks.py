
# adapted from https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook
import torch
import torch.nn as nn

# ConvNeXt PAPER: https://arxiv.org/pdf/2201.03545
# ConvNeXt CODE: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# Slightly modified: 2x expansion in pointwise conv instead of 4x, and LayerNorm over (F, C) instead of just C
class ConvNeXtBlockMPECL(nn.Module):
    """ConvNeXt block for (B, C, F, T) inputs, computed channel-last.

    Internally permutes to (B, T, F, C) so that nn.LayerNorm normalizes over
    (F, C) for each time step and nn.Linear operates on the channel dimension.
    """
    def __init__(self, in_channels, in_bins, out_channels, drop_path=0.0, k_t=7, k_f=7, **kwargs):
        super().__init__()

        # 1. Depthwise Conv (B, C, F, T)
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=(k_f, k_t),
                                padding=(k_f // 2, k_t // 2), groups=in_channels)

        # 2. Norm over (F, C), applied in channel-last layout (B, T, F, C)
        # Note: elementwise_affine=False to avoid learning scale and bias parameters
        self.norm = nn.LayerNorm([in_bins, in_channels], eps=1e-6, elementwise_affine=False)

        # 3. Pointwise ops via Linear with 2x expansion (instead of 4x in original ConvNeXt paper)
        self.pwconv1 = nn.Linear(in_channels, 2 * in_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * in_channels, out_channels)

        self.drop_path = nn.Identity() if drop_path == 0 else StochasticDepth(drop_path)

    def forward(self, x):
        # x: (B, C, F, T)
        residual = x
        x = self.dwconv(x)

        x = x.permute(0, 3, 2, 1)   # (B, C, F, T) -> (B, T, F, C)
        x = self.norm(x)
        x = self.pwconv1(x)          # (B, T, F, C) -> (B, T, F, 2C)
        x = self.act(x)
        x = self.pwconv2(x)          # (B, T, F, 2C) -> (B, T, F, C_out)
        x = x.permute(0, 3, 2, 1)   # (B, T, F, C_out) -> (B, C_out, F, T)

        return self.drop_path(x) + residual

# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
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
