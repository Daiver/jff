import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

import np_draw_tools

import torch_tools
from image2pointsdataset import Image2PointsDataset

from generate_data import generate_dataset, generate_dataset_and_compute_flow
from model import Model


def main():
    device = 'cuda:0'
    batch_size = 64

    model_super = Model().to(device)
    model_super.load_state_dict(torch.load("checkpoints/checkpoint_supervised.pt"))
    model_super.eval()

    model_weak = Model().to(device)
    model_weak.load_state_dict(torch.load("checkpoints/checkpoint_weaksupervised.pt"))
    model_weak.eval()

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

        predict_weak = model_weak(images)
        predict_weak = torch_tools.norm_to_screen(predict_weak, images.shape[3], images.shape[2])

        predict_super = model_super(images)
        predict_super = torch_tools.norm_to_screen(predict_super, images.shape[3], images.shape[2])

        for i in range(images.shape[0]):
            image = images[i].detach().permute(1, 2, 0).cpu().numpy().copy()
            predict_weak_one = predict_weak[i].detach().cpu().numpy()
            predict_super_one = predict_super[i].detach().cpu().numpy()

            cv2.circle(image, np_draw_tools.to_int_tuple(predict_weak_one), 2, color=(0, 0, 255), thickness=-1)
            cv2.circle(image, np_draw_tools.to_int_tuple(predict_super_one), 2, color=(255, 255, 0), thickness=-1)

            cv2.imshow("img", image)
            cv2.waitKey()


if __name__ == '__main__':
    main()
