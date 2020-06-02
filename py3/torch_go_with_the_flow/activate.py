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

from train import Model, generate_dataset_and_compute_flow, generate_dataset


def main():
    device = 'cuda:0'
    batch_size = 64

    model = Model().to(device)
    # model.load_state_dict(torch.load("checkpoints/checkpoint_supervised.pt"))
    model.load_state_dict(torch.load("checkpoints/checkpoint_weaksupervised.pt"))
    model.eval()

    train_images, train_positions = generate_dataset()
    # visualize_dataset(train_images, train_positions, train_flows)
    train_points_dataset_full = Image2PointsDataset(train_images, train_positions)

    supervised_indices = list(range(0, len(train_images), 50))
    print(supervised_indices)
    train_points_dataset = Image2PointsDataset(
        np.array(train_images)[supervised_indices],
        np.array(train_positions)[supervised_indices]
    )

    val_points_loader = DataLoader(train_points_dataset_full, batch_size=batch_size, shuffle=False)
    train_points_loader = DataLoader(train_points_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    for iter_ind, batch_data in enumerate(val_points_loader):
        images, positions = batch_data
        images, positions = images.to(device), positions.to(device)

        predict = model(images)
        predict = torch_tools.norm_to_screen(predict, images.shape[3], images.shape[2])
        for i in range(images.shape[0]):
            image = images[i].detach().permute(1, 2, 0).cpu().numpy().copy()
            predict_one = predict[i].detach().cpu().numpy()
            print(image.dtype)
            cv2.circle(image, np_draw_tools.to_int_tuple(predict_one), 2, color=(0, 0, 255), thickness=-1)

            cv2.imshow("img", image)
            cv2.waitKey()


if __name__ == '__main__':
    main()
