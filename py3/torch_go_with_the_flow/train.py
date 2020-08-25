import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import np_draw_tools

from generate_data import generate_dataset_and_compute_flow
from model import Model
from utils import visualize_optical_flow, warp_by_flow
import torch_tools
from image2pointsdataset import Image2PointsDataset
from clip2flowdataset import Clip2FlowDataset


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


def main():
    device = 'cuda:0'
    batch_size = 64
    # batch_size = 128
    lr = 1e-5
    n_epochs = 5000

    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    train_images, train_positions, train_flows = generate_dataset_and_compute_flow()
    # visualize_dataset(train_images, train_positions, train_flows)
    train_points_dataset_full = Image2PointsDataset(train_images, train_positions)

    # supervised_indices = list(range(0, len(train_images), 50))
    # supervised_indices = list(range(0, len(train_images), 30))
    supervised_indices = list(range(0, len(train_images), 20))

    print(supervised_indices)
    train_points_dataset = Image2PointsDataset(
        np.array(train_images)[supervised_indices * 10],
        np.array(train_positions)[supervised_indices * 10]
    )
    train_flows_dataset = Clip2FlowDataset(train_images, train_flows)

    val_points_loader = DataLoader(train_points_dataset_full, batch_size=batch_size, shuffle=True)
    train_points_loader = DataLoader(train_points_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_flows_loader = DataLoader(train_flows_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print("len(train_points_dataset)", len(train_points_dataset))
    print("len(train_flows_dataset)", len(train_flows_dataset))

    for epoch in range(1000):
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

    torch.save(model.state_dict(), "checkpoints/checkpoint_supervised.pt")

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
            flow_values = torch_tools.screen_to_norm(flow_values, images0.shape[3], images0.shape[2]) + 1

            loss = criterion(diff, flow_values)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            # print("Additional supervised step")
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
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "checkpoints/checkpoint_weaksupervised.pt")


if __name__ == '__main__':
    main()


# import torch
