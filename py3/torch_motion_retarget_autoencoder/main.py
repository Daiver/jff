import numpy as np
import cv2

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import np_draw_tools

from models import Encoder, Decoder

canvas_size = (128, 128)


def draw_circle_sample(center):
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    point = np.array(center).round().astype(np.int32)
    cv2.circle(canvas, (point[0], point[1]), 5, (0, 255, 0), thickness=-1)
    return canvas


def make_circle_dataset():
    res = []
    for y in range(10, 110, 2):
        for x in range(10, 110, 2):
            res.append(draw_circle_sample((x, y)))
    return res


def draw_rectangle_sample(center):
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    point = np.array(center).round().astype(np.int32)
    pt1 = (point[0] - 5, point[1] - 5)
    pt2 = (point[0] + 5, point[1] + 5)
    cv2.rectangle(canvas, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=-1)
    return canvas


def make_rectangle_dataset():
    res = []
    for y in range(10, 110, 2):
        for x in range(10, 110, 2):
            res.append(draw_rectangle_sample((x, y)))
    return res


def main():
    device = "cuda"
    epochs = 50
    batch_size = 8

    circle_set = make_circle_dataset()
    rect_set = make_rectangle_dataset()
    # for i, x in enumerate(circle_set):
    #     cv2.imshow("", x)
    #     cv2.waitKey()

    train_rect_set = [
        torch.from_numpy(x).float().permute(2, 0, 1) / 255.0
        for x in rect_set
    ]

    train_circle_set = [
        torch.from_numpy(x).float().permute(2, 0, 1) / 255.0
        for x in circle_set
    ]

    print(f"N samples {len(train_rect_set)}")
    train_rect_loader = DataLoader(train_rect_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_rect_loader = DataLoader(train_rect_set, batch_size=batch_size * 4, shuffle=False)

    train_circle_loader = DataLoader(train_circle_set, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder = Encoder().to(device)
    decoder_rect = Decoder().to(device)
    decoder_circle = Decoder().to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder_rect.parameters()) + list(decoder_circle.parameters()), lr=1e-4)
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    criterion = nn.BCELoss()
    for epoch_ind in range(epochs):
        losses = []
        losses_rect = []
        losses_circle = []

        encoder.train()
        decoder_rect.train()
        decoder_circle.train()

        for sample_rect, sample_circle in zip(train_rect_loader, train_circle_loader):

            sample_rect = sample_rect.to(device)
            pred_rect = encoder(sample_rect)
            pred_rect = decoder_rect(pred_rect)
            loss_rect = criterion(pred_rect, sample_rect)

            sample_circle = sample_circle.to(device)
            pred_circle = encoder(sample_circle)
            pred_circle = decoder_circle(pred_circle)
            loss_circle = criterion(pred_circle, sample_circle)

            loss = loss_rect + loss_circle

            losses.append(loss.item())
            losses_rect.append(loss_rect.item())
            losses_circle.append(loss_circle.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        encoder.eval()
        decoder_rect.eval()
        decoder_circle.eval()

        print(f"{epoch_ind + 1} / {epochs} loss {np.mean(losses)} loss rect {np.mean(losses_rect)} loss circle {np.mean(losses_circle)}")
        for sample_rect in val_rect_loader:
            sample_rect = sample_rect.to(device)
            pred_rect = encoder(sample_rect)
            # pred_rect = decoder_rect(pred_rect)
            pred_rect = decoder_circle(pred_rect)
            pred_rect = (pred_rect.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            sample_rect = (sample_rect.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            to_show = []
            for p, s in zip(pred_rect, sample_rect):
                to_show.append(p)
                to_show.append(s)
            to_show = np_draw_tools.make_grid(to_show[:32], 4)
            cv2.imshow("", to_show)
            cv2.waitKey(100)
            break
    cv2.waitKey()


if __name__ == '__main__':
    main()

