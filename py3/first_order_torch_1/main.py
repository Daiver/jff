from typing import Union, List, Tuple
import random
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import np_draw_tools


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def draw_background(canvas_size: Union[List[int], Tuple[int, int]]):
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    # scale = canvas_size[0] / 64.0 * 1.5
    # cv2.putText(
    #     canvas, "A",
    #     (canvas_size[0] // 2, canvas_size[1] // 2),
    #     cv2.FONT_HERSHEY_COMPLEX, scale, (0.5, 0.5, 0.5))
    #
    # cv2.putText(
    #     canvas, "B",
    #     (canvas_size[0] // 2 - canvas_size[0] // 3, canvas_size[1] // 2 + canvas_size[1] // 4),
    #     cv2.FONT_HERSHEY_COMPLEX, scale, (0.0, 0.7, 0.5))
    return canvas


def draw_fig1_on_background(background: np.ndarray, pos) -> np.ndarray:
    canvas = background.copy()
    points = np.array([
        [30, 30],
        [30, 50],
        [50, 30]
    ], dtype=np.float32)
    points += pos
    points = points.round().astype(np.int32).reshape((1, -1, 2))
    cv2.fillPoly(canvas, points, color=(0, 0, 1))
    return canvas


def mk_translation_sequence(background: np.ndarray, n_samples: int, samples_maker) -> np.ndarray:
    random_translations = np.random.uniform(-10, 10, size=(n_samples, 2))
    return np.array([
        samples_maker(background, pos)
        for pos in random_translations
    ], dtype=np.float32)


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

    def forward(self, x):
        z = self.strum(x)
        if self.shortcut is not None:
            x = self.shortcut(x)

        return x + z


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        n_feature_channels = 16
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, n_feature_channels, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(n_feature_channels),
            nn.LeakyReLU()
        )
        self.block1 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)
        self.block2 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)
        self.block3 = ResidualBlock(in_channels=n_feature_channels, out_channels=n_feature_channels, stride=2)

        n_fc_in = n_feature_channels * 4 * 4
        self.fc = nn.Linear(n_fc_in, 2)

    def forward(self, x):
        x = self.first_conv(x)  # 64 -> 32
        x = self.block1(x)      # 32 -> 16
        x = self.block2(x)      # 16 -> 8
        x = self.block3(x)      # 8  -> 4

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def main():
    canvas_size = (64, 64)
    n_samples = 200
    device = "cuda"
    batch_size = 32
    n_epochs = 10

    background = draw_background(canvas_size)
    fig1 = draw_fig1_on_background(background, [10, 10])
    figs = mk_translation_sequence(background, n_samples, draw_fig1_on_background)

    grid = np_draw_tools.make_grid(figs[:50], 4)

    cv2.imshow("back", background)
    cv2.imshow("fig1", grid)
    cv2.waitKey(100)

    dataset = torch.from_numpy(figs).permute([0, 3, 1, 2])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Model().to(device)

    for epoch_ind in range(n_epochs):
        for batch_ind, batch in enumerate(dataloader):
            n_real_samples = batch.shape[0]


    ans = model(dataset.to(device))



if __name__ == '__main__':
    main()

