import random
import sys
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import pretrainedmodels
import albumentations

from datasets import mk_kostet_dataset
from train_utils import run_validate
from models import Model
import paths


def mk_img_mesh_transforms(image_transforms):

    def apply(img, mesh):
        return image_transforms(img), torch.from_numpy(mesh)
    return apply


def main():

    epochs = 1000
    batch_size = 128
    lr = 0.02
    device = 'cuda'

    backbone = pretrainedmodels.resnet18()
    model = Model(backbone, 512, 9591)

    train_img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(backbone.mean, backbone.std),
    ])
    val_img_transforms = train_img_transforms
    train_transforms = mk_img_mesh_transforms(train_img_transforms)
    val_transforms = mk_img_mesh_transforms(val_img_transforms)

    train_dataset = mk_kostet_dataset(list(range(0, 100)) + list(range(200, 299)), train_transforms)
    val_dataset = mk_kostet_dataset(list(range(100, 200)), val_transforms)

    print(f"n train = {len(train_dataset)}, n val = {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # model = torch.load("checkpoints/100_0.6407536556351472.pt")

    print("START TRAINING")
    model.to(device)
    losses = []
    for epoch in range(epochs):

        model.train()
        for inp, target in train_loader:
            inp, target = inp.to(device), target.to(device)
            output = model(inp)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
        model.eval()

        mean_loss = np.mean(losses)
        val_loss = run_validate(model, criterion, val_loader, device)
        print(f"{epoch + 1}/{epochs} loss = {mean_loss}, val_loss = {val_loss}")
        torch.save(model.state_dict(), f"checkpoints/{epoch}_{mean_loss}_{val_loss}.pt")


if __name__ == '__main__':
    main()