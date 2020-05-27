import time
import pickle
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import np_draw_tools

import utils
from utils import visualize_optical_flow, warp_by_flow
import torch_tools
from image2pointsdataset import Image2PointsDataset
from clip2flowdataset import Clip2FlowDataset

canvas_size = (128, 128)


def draw_sample(coord):
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
    x, y = np.array(coord).round().astype(np.int32)
    cv2.circle(canvas, center=(x, y), radius=10, color=(0, 255, 0), thickness=-1)
    return canvas


def generate_dataset():
    n_x_steps = 80
    # n_x_steps = 2
    n_y_steps = 3

    point_orig = [25, 50]
    images = []
    positions = []
    for dy in range(n_y_steps):
        for dx in range(n_x_steps):
            point = [point_orig[0] + dx, point_orig[1] + 5 * dy]
            positions.append(point)
            images.append(draw_sample(point))

    return images, positions


def generate_dataset_and_compute_flow():
    images, positions = generate_dataset()

    time_start = time.time()
    flows = utils.compute_flow_for_clip(images)
    print(f"elapsed {time.time() - time_start}")
    return images, positions, flows


def visualize_dataset(images, positions, flows):
    for i in range(len(images) - 1):
        img0 = images[i]
        img1 = images[i + 1]
        flow = flows[i]
        warp = warp_by_flow(img1, flow)

        flow_vis_rgb = visualize_optical_flow(flow)

        cv2.imshow("i0_w", warp)
        cv2.imshow("img0", img0)
        cv2.imshow("img1", img1)
        cv2.imshow("flow", flow_vis_rgb)
        np_draw_tools.wait_esc()


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
        n_feats = 16

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
            nn.BatchNorm1d(n_feats),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=n_feats, out_features=n_outs),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, output_size=(4, 4)).view(batch_size, -1)
        x = self.head(x)
        return x


def main():
    device = 'cuda:0'
    batch_size = 16
    lr = 1e-4
    n_epochs = 1000

    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    train_images, train_positions, train_flows = generate_dataset_and_compute_flow()
    # visualize_dataset(train_images, train_positions, train_flows)
    train_points_dataset_full = Image2PointsDataset(train_images, train_positions)
    train_points_dataset = Image2PointsDataset(train_images[:5], train_positions[:5])
    train_flows_dataset = Clip2FlowDataset(train_images, train_flows)

    val_points_loader = DataLoader(train_points_dataset_full, batch_size=batch_size, shuffle=True)
    train_points_loader = DataLoader(train_points_dataset, batch_size=batch_size, shuffle=True)
    train_flows_loader = DataLoader(train_flows_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(100):
        model.train()
        losses = []
        for iter_ind, batch_data in enumerate(train_points_loader):
            images, positions = batch_data
            images, positions = images.to(device), positions.to(device)
            positions = torch_tools.screen_to_norm(positions, images.shape[3], images.shape[2])

            predict = model(images)
            loss = criterion(predict, positions)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        val_losses = []
        for iter_ind, batch_data in enumerate(val_points_loader):
            with torch.no_grad():
                images, positions = batch_data
                images, positions = images.to(device), positions.to(device)
                positions = torch_tools.screen_to_norm(positions, images.shape[3], images.shape[2])

                predict = model(images)
                loss = criterion(predict, positions)
                val_losses.append(loss.item())

        print(f"{epoch + 1}/{n_epochs}: loss {np.mean(losses)}, val_loss {np.mean(val_losses)}")

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for iter_ind, batch_data in enumerate(train_flows_loader):
            images0, images1, flows = batch_data
            images0, images1, flows = images0.to(device), images1.to(device), flows.to(device)

            predict0 = model(images0)
            predict1 = model(images1)

            diff = predict1 - predict0
            flow_values = F.grid_sample(flows, predict0.view(-1, 1, 1, 2), align_corners=True).view(-1, 2)

            loss = criterion(diff, flow_values)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        val_losses = []
        for iter_ind, batch_data in enumerate(val_points_loader):
            with torch.no_grad():
                images, positions = batch_data
                images, positions = images.to(device), positions.to(device)
                positions = torch_tools.screen_to_norm(positions, images.shape[3], images.shape[2])

                predict = model(images)
                loss = criterion(predict, positions)
                val_losses.append(loss.item())

        print(f"{epoch + 1}/{n_epochs}: loss {np.mean(losses)}, val_loss {np.mean(val_losses)}")


if __name__ == '__main__':
    main()


# import torch
