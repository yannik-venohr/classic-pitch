import torch.nn as nn
from src.model.building_blocks import ConvNeXtBlockMPECL


class HCQTEncoder(nn.Module):
    """
    Multi-pitch encoder of ClassicPitch.
    Input shape: (batch_size, harmonics, freq, time)
    Args:
        in_bins:         Number of input frequency bins (e.g. 216 for 6 octaves with 36 bins each)
        in_channels:     Number of input channels (harmonics in HCQT)
        out_channels:    Number of output channels (1 for pitch)
        widths:          Channel widths of the two stages
        depths:          Number of ConvNeXt blocks of the two stages
        k_f, k_t:        Kernel sizes (frequency, time) of the two stages
        drop_path_rate:  Maximum stochastic depth rate, increases linearly over the blocks
    """
    def __init__(self, in_bins=216, in_channels=6, out_channels=1, widths=[40, 80], depths=[2, 8], k_f=[7, 13], k_t=[7, 7], drop_path_rate=0.2, **kwargs):
        super(HCQTEncoder, self).__init__()

        total_depth = sum(depths)
        dp_rates = [drop_path_rate * i / (total_depth - 1) for i in range(total_depth)]

        # 1. Stem
        self.stem = nn.Conv2d(in_channels, widths[0], kernel_size=1)

        # 2. Stage 1 at 3 bins per semitone
        self.stage1 = nn.ModuleList([
            ConvNeXtBlockMPECL(in_channels=widths[0],
                               in_bins=in_bins,
                               out_channels=widths[0],
                               k_t=k_t[0],
                               k_f=k_f[0],
                               drop_path=dp_rates[i])
            for i in range(depths[0])
        ])

        # 3. Downsampling to 1 bin per semitone and increasing channels
        self.downsample = nn.Conv2d(widths[0], widths[1], kernel_size=(3, 1), stride=(3, 1))

        # 4. Stage 2 at 1 bin per semitone
        self.stage2 = nn.ModuleList([
            ConvNeXtBlockMPECL(in_channels=widths[1],
                               in_bins=in_bins // 3,
                               out_channels=widths[1],
                               k_t=k_t[1],
                               k_f=k_f[1],
                               drop_path=dp_rates[i + depths[0]])
            for i in range(depths[1])
        ])

        # 5. Classification Head
        self.head = nn.Conv2d(widths[1], out_channels, kernel_size=1)

    def forward_features(self, x):
        """Return features before the classification head."""
        x = self.stem(x)
        for layer in self.stage1:
            x = layer(x)
        x = self.downsample(x)
        for layer in self.stage2:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)
