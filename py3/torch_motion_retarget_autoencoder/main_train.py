import numpy as np
import cv2

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import np_draw_tools

from data_generation import make_rectangle_dataset, make_circle_dataset
from models import Encoder, Decoder
from utils import numpy_images_to_torch


def main():
    n_samples_to_generate = 1500

    epochs = 50
    device = "cuda"
    batch_size = 8

    circle_set = make_circle_dataset(n_samples_to_generate)
    rect_set = make_rectangle_dataset(n_samples_to_generate)

    train_rect_set = numpy_images_to_torch(rect_set)
    train_circle_set = numpy_images_to_torch(circle_set)

    print(f"N rect samples {len(train_rect_set)} N circle samples {len(train_circle_set)}")
    train_rect_loader = DataLoader(train_rect_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_rect_loader = DataLoader(train_rect_set, batch_size=batch_size * 16, shuffle=False)

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
            to_show = np_draw_tools.make_grid(to_show[:64], 8)
            cv2.imshow("", to_show)
            cv2.waitKey(100)
            break
    torch.save(encoder.state_dict(), "encoder.pt")
    torch.save(decoder_circle.state_dict(), "decoder_circle.pt")
    torch.save(decoder_rect.state_dict(), "decoder_rect.pt")
    cv2.waitKey()


if __name__ == '__main__':
    main()

