import numpy as np
import cv2

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import np_draw_tools

from model import Model

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


def main():
    device = "cuda"
    epochs = 20
    batch_size = 64

    circle_set = make_circle_dataset()
    # for i, x in enumerate(circle_set):
    #     cv2.imshow("", x)
    #     cv2.waitKey()

    train_set = [
        torch.from_numpy(x).float().permute(2, 0, 1) / 255.0
        for x in circle_set
    ]
    print(f"N samples {len(train_set)}")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    # criterion = nn.BCELoss()
    for epoch_ind in range(epochs):
        losses = []
        model.train()
        for sample in train_loader:
            sample = sample.to(device)
            pred = model(sample)
            loss = criterion(pred, sample)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        print(f"{epoch_ind + 1} / {epochs} loss {np.mean(losses)}")
        for sample in val_loader:
            sample = sample.to(device)
            pred = model(sample)
            pred = (pred.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            sample = (sample.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            to_show = []
            for p, s in zip(pred, sample):
                to_show.append(p)
                to_show.append(s)
            to_show = np_draw_tools.make_grid(to_show, 2)
            cv2.imshow("", to_show)
            cv2.waitKey(100)
            break


if __name__ == '__main__':
    main()

