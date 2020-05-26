import time
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import np_draw_tools

import utils
from utils import visualize_optical_flow, warp_by_flow

canvas_size = (128, 128)


def draw_sample(coord):
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
    x, y = np.array(coord).round().astype(np.int32)
    cv2.circle(canvas, center=(x, y), radius=10, color=(0, 255, 0), thickness=-1)
    return canvas


def generate_dataset():
    n_samples = 80

    point = [25, 50]
    images = []
    positions = []
    for i in range(n_samples):
        positions.append(point)
        images.append(draw_sample(point))
        point[0] += 1

    return images, positions


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


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        n_outs = 2
        n_feats = 32

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(inplace=True),

            ResidualBlock(in_channels=n_feats, out_channels=n_feats, stride=2),  # 128 -> 64
            ResidualBlock(in_channels=n_feats, out_channels=n_feats, stride=2),  # 64 -> 32
            ResidualBlock(in_channels=n_feats, out_channels=n_feats, stride=2),  # 32 -> 16
            ResidualBlock(in_channels=n_feats, out_channels=n_feats, stride=2),  # 16 -> 8
            ResidualBlock(in_channels=n_feats, out_channels=n_feats, stride=2),  # 8 -> 4
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=4*4*n_feats, out_features=n_feats),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=n_feats, out_features=n_outs),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, output_size=(4, 4))
        x = self.head(x)
        return x


def main():
    train_images, train_positions = generate_dataset()

    time_start = time.time()
    flows = utils.compute_flow_for_clip(train_images)
    print(f"elapsed {time.time() - time_start}")
    # for i in range(len(train_images) - 1):
    #     img0 = train_images[i]
    #     img1 = train_images[i + 1]
    #     flow = flows[i]
    #     warp = warp_by_flow(img1, flow)
    #
    #     flow_vis_rgb = visualize_optical_flow(flow)
    #
    #     cv2.imshow("i0_w", warp)
    #     cv2.imshow("img0", img0)
    #     cv2.imshow("img1", img1)
    #     cv2.imshow("flow", flow_vis_rgb)
    #     np_draw_tools.wait_esc()


if __name__ == '__main__':
    main()


# import torch
