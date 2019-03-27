import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordConv2d(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias):
        super().__init__()
        print("WARNING: Not tested at all")
        self.conv = nn.Conv2d(
            in_channels=in_channels + 2, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.coord_feature_map = None
        assert False

    def init_coord_feature_map(self, required_rows_cols):
        if self.coord_feature_map is not None:
            assert self.coord_feature_map.shape[2] == required_rows_cols[0]
            assert self.coord_feature_map.shape[3] == required_rows_cols[1]
            return

        theta = torch.FloatTensor([1, 0, 0, 0, 1, 0]).view(1, 2, 3)
        self.coord_feature_map = \
            F.affine_grid(theta, (1, 1, required_rows_cols[0], required_rows_cols[1])).transpose(1, 3)

    def forward(self, x):
        self.init_coord_feature_map((x.shape[2], x.shape[3]))
        x = torch.cat((x, self.coord_feature_map), dim=1)
        return self.conv(x)
