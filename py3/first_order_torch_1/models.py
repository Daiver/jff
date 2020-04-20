from torch import nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        assert stride == 1 or stride == 2
        super().__init__()
        self.strum = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
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


class LinkNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int = 1):
        super().__init__()
        assert scale == 1 or scale == 2
        super().__init__()
        self.strum = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale),
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

        if out_channels != in_channels or scale != 1:
            self.shortcut = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        z = self.strum(x)
        if self.shortcut is not None:
            x = self.shortcut(x)

        return x + z


class KpPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        n_feature_channels = 32
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, n_feature_channels, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(n_feature_channels),
            nn.LeakyReLU()
        )
        self.block1 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)
        self.block2 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)
        self.block3 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)
        self.block4 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)

        n_fc_in = n_feature_channels * 4 * 4
        self.fc = nn.Linear(n_fc_in, 2)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()

    def forward(self, x):
        x = self.first_conv(x)  # 64 -> 64
        x = self.block1(x)      # 64 -> 32
        x = self.block2(x)      # 32 -> 16
        x = self.block3(x)      # 16 -> 8
        x = self.block4(x)      # 8  -> 4

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class Hourglass(nn.Module):
    def __init__(
            self,
            n_in_channels,
            n_out_channels,
            n_feature_channels=16
    ):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(n_in_channels, n_feature_channels, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(n_feature_channels),
            nn.LeakyReLU()
        )
        self.down_block1 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)
        self.down_block2 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)
        self.down_block3 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)
        self.down_block4 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)

        self.up_block4 = LinkNetBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, scale=2)
        self.up_block3 = LinkNetBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, scale=2)
        self.up_block2 = LinkNetBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, scale=2)
        self.up_block1 = LinkNetBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, scale=2)

        self.final = nn.Conv2d(
            in_channels=n_feature_channels, out_channels=n_out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        first = self.first_conv(x)  # 64 -> 64

        down1 = self.down_block1(first)  # 64 -> 32
        down2 = self.down_block2(down1)  # 32 -> 16
        down3 = self.down_block3(down2)  # 16 -> 8
        down4 = self.down_block4(down3)  # 8  -> 4

        up4 = self.up_block4(down4) + down3  # 4  -> 8
        up3 = self.up_block3(up4) + down2    # 8  -> 16
        up2 = self.up_block2(up3) + down1    # 16 -> 32
        up1 = self.up_block1(up2) + first    # 32 -> 64

        return self.final(up1)
