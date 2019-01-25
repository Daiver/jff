import torch.nn.functional as F
import torch.nn as nn


class SegNet1(nn.Module):
    def __init__(self):
        super(SegNet1, self).__init__()
        nStFilters = 256
        self.convD1 = nn.Conv2d(3 , nStFilters, kernel_size=3, padding=1)
        self.convD2 = nn.Conv2d(nStFilters, 2*nStFilters, kernel_size=3, padding=1)
        self.convD3 = nn.Conv2d(2*nStFilters, 2*nStFilters, kernel_size=3, padding=1)

        self.convE1 = nn.Conv2d(2*nStFilters, 2*nStFilters, kernel_size=3, padding=1)
        self.convE2 = nn.Conv2d(2*nStFilters, 2*nStFilters, kernel_size=3, padding=1)
        self.convE3 = nn.Conv2d(2*nStFilters, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convD1(x), 2))
        x = F.relu(F.max_pool2d(self.convD2(x), 2))
        x = F.relu(F.max_pool2d(self.convD3(x), 2))
        x = F.upsample(x, scale_factor=2)
        x = F.relu(self.convE1(x))
        x = F.upsample(x, scale_factor=2)
        x = F.relu(self.convE2(x))
        x = F.upsample(x, scale_factor=2)
        x = F.relu(self.convE3(x))
        return x

