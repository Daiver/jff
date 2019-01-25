import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=0, bias=False), # 30
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 3, stride=2, padding=0, bias=False), # 13
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 3, stride=1, padding=0, bias=False), # 4
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 3, stride=1, padding=0, bias=False), # 1?
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.fc_final = nn.Linear(32, 2)
        self.fc_final.weight.data.zero_()

    def forward(self, x):
        res = self.feature_extractor(x)
        res = F.adaptive_avg_pool2d(res, (1, 1))
        res = res.view(-1, 32)
        res = self.fc_final(res)

        return res
