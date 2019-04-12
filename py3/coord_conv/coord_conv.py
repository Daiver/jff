import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordConv2d(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels + 2, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.coord_feature_map = None

    @staticmethod
    def device_from_tensor(tensor: torch.Tensor):
        if not tensor.is_cuda:
            return "cpu"
        return f"cuda:{tensor.get_device()}"

    def init_coord_feature_map(self, required_batch_rows_cols, device):
        if self.coord_feature_map is not None:
            if (self.coord_feature_map.shape[0] == required_batch_rows_cols[0] and
                    self.coord_feature_map.shape[2] == required_batch_rows_cols[1] and
                    self.coord_feature_map.shape[3] == required_batch_rows_cols[2]):
                return

        theta = torch.FloatTensor([1, 0, 0, 0, 1, 0]).view(1, 2, 3)
        theta = torch.cat((theta, ) * required_batch_rows_cols[0], dim=0)

        self.coord_feature_map = \
            F.affine_grid(theta,
                          (required_batch_rows_cols[0], 1, required_batch_rows_cols[1], required_batch_rows_cols[2])
                          ).transpose(1, 3).to(device)

    def forward(self, x):
        self.init_coord_feature_map((x.shape[0], x.shape[2], x.shape[3]), device=self.device_from_tensor(x))
        # print(f"x.shape {x.shape}, self.coord.shape {self.coord_feature_map.shape}")
        x = torch.cat((self.coord_feature_map, x), dim=1)
        return self.conv(x)
