import torch.nn as nn

from src.model.building_blocks import ConvNeXtBlockMPECL

class OnsetModule(nn.Module):
    """
    Onset detection module.
    Input:  (B, C_in, F, T) — output of the second HCQT Encoder
    Output: (B, 1, F, T)    — onset logits
    """
    def __init__(self, in_channels=41, in_bins=72, width=40, depth=2, k_t=15, k_f=5, drop_path_rate=0.1, secondary_dim=5, norm="CF", channel_last=False, simple_stem=False, simple_head=False, use_diff=True, **kwargs):
        super().__init__()
        print(f"Initializing OnsetModule with in_channels={in_channels}, in_bins={in_bins}, width={width}, depth={depth}, k_t={k_t}, k_f={k_f}, drop_path_rate={drop_path_rate}, secondary_dim={secondary_dim}, norm={norm}")

        self.stem =nn.Conv2d(in_channels, width, kernel_size=1)
        
        # 2. ConvNeXt blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlockMPECL(
                in_channels=width,
                in_bins=in_bins,
                out_channels=width,
                k_t=k_t,
                k_f=k_f,
                drop_path=drop_path_rate * (i + 1) / depth,
                norm=norm
            )
            for i in range(depth)
        ])
        
        # 3. Onset head → (B, 1, F, T)
        self.head = nn.Conv2d(width, 1, kernel_size=1)
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x) 