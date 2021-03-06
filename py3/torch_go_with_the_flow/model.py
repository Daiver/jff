from torch import nn as nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        n_outs = 2
        n_feats = 16

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(inplace=True),

            ResidualBlock(in_channels=n_feats, out_channels=2*n_feats, stride=2),  # 128 -> 64
            ResidualBlock(in_channels=2*n_feats, out_channels=2*n_feats, stride=2),  # 64 -> 32
            ResidualBlock(in_channels=2*n_feats, out_channels=2*n_feats, stride=2),  # 32 -> 16
            ResidualBlock(in_channels=2*n_feats, out_channels=2*n_feats, stride=2),  # 16 -> 8
            # ResidualBlock(in_channels=2*n_feats, out_channels=2*n_feats, stride=2),  # 8 -> 4
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 2 * n_feats, out_features=n_outs),
            # nn.Linear(in_features=4*4*2*n_feats, out_features=2*n_feats),
            # nn.BatchNorm1d(2*n_feats),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(in_features=2*n_feats, out_features=n_outs),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, output_size=(4, 4)).view(batch_size, -1)
        x = self.head(x)
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
