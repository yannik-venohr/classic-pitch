import torch.nn as nn

from src.model.building_blocks import ConvNeXtBlockMPECL


class OnsetModule(nn.Module):
    """
    Onset detection module.
    Input:  (B, C_in, F, T) — features of the HCQTEncoder
    Output: (B, 1, F, T)    — onset logits
    """
    def __init__(self, in_channels=80, in_bins=72, width=40, depth=2, k_t=7, k_f=7, drop_path_rate=0.1, **kwargs):
        super().__init__()

        self.stem = nn.Conv2d(in_channels, width, kernel_size=1)

        self.blocks = nn.ModuleList([
            ConvNeXtBlockMPECL(
                in_channels=width,
                in_bins=in_bins,
                out_channels=width,
                k_t=k_t,
                k_f=k_f,
                drop_path=drop_path_rate * (i + 1) / depth,
            )
            for i in range(depth)
        ])

        self.head = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
