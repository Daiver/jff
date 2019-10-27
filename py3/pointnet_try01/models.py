import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, middle_channels):
        super().__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(True),

            nn.Conv1d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=1),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(True)
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=middle_channels, out_features=middle_channels),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(True),
            nn.Linear(in_features=middle_channels, out_features=middle_channels),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(True),
            nn.Linear(in_features=middle_channels, out_features=in_channels * in_channels)

        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = x.view(batch_size, self.middle_channels)
        x = self.head(x)

        x = x.view(-1, self.in_channels, self.in_channels)
        eyes = np.eye(self.in_channels, dtype=np.float32)
        eyes = torch.from_numpy(eyes)\
            .flatten()\
            .repeat(x.shape[0], 1)\
            .view(-1, self.in_channels, self.in_channels)\
            .to(x.device)
        return x + eyes


class PointNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        middle_channels = 64
        out_channels = 1
        self.first_transformer = SpatialTransformer(in_channels=in_channels, middle_channels=middle_channels)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(True),

            nn.Conv1d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=1),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(True)
        )
        # self.final_transformer = SpatialTransformer(in_channels=middle_channels, middle_channels=middle_channels)
        self.head = nn.Sequential(
            nn.Linear(in_features=middle_channels, out_features=middle_channels),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(True),
            nn.Linear(in_features=middle_channels, out_features=out_channels),
        )

    def forward(self, x):
        print(f"inp = {x.shape}")
        first_trans = self.first_transformer(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, first_trans)
        x = x.transpose(2, 1)
        print(f"after transform = {x.shape}")

        x = self.feature_extractor(x)
        print(f"after feature = {x.shape}")

        x = torch.max(x, dim=2, keepdim=True)[0]
        x = x.view(x.shape[0], -1)
        x = self.head(x)

        return x
