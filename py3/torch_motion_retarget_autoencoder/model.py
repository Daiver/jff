from torch import nn as nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        n_outs = 2
        n_feats = 32

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(inplace=True),

            ResidualBlock(in_channels=n_feats, out_channels=n_feats, stride=2),  # 128 -> 64
            ResidualBlock(in_channels=n_feats, out_channels=n_feats, stride=2),  # 64 -> 32
            ResidualBlock(in_channels=n_feats, out_channels=n_feats, stride=2),  # 32 -> 16
            ResidualBlock(in_channels=n_feats, out_channels=2 * n_feats, stride=2),  # 16 -> 8
            ResidualBlock(in_channels=2 * n_feats, out_channels=4 * n_feats, stride=2),  # 8 -> 4
            ResidualBlock(in_channels=4 * n_feats, out_channels=n_feats, stride=1),  # 4 -> 4
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=1, padding=0, bias=True),
            ResidualBlock(in_channels=n_feats, out_channels=16 * n_feats, stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 4 -> 8
            ResidualBlock(in_channels=16 * n_feats, out_channels=8 * n_feats, stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 8 -> 16
            ResidualBlock(in_channels=8 * n_feats, out_channels=4 * n_feats, stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 16 -> 32
            ResidualBlock(in_channels=4 * n_feats, out_channels=2 * n_feats, stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 32 -> 64
            ResidualBlock(in_channels=2 * n_feats, out_channels=n_feats, stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 64 -> 128
            ResidualBlock(in_channels=n_feats, out_channels=n_feats, stride=1),
            nn.Conv2d(in_channels=n_feats, out_channels=3, kernel_size=1, padding=0, bias=True),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        assert stride == 1 or stride == 2
        super().__init__()
        self.strum = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

        if out_channels != in_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        else:
            self.shortcut = None

        for x in self.modules():
            if isinstance(x, nn.BatchNorm2d):
                x.bias.data.zero_()
                x.weight.data.zero_()

    def forward(self, x):
        z = self.strum(x)
        if self.shortcut is not None:
            x = self.shortcut(x)

        return x + z