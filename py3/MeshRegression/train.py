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
from models import Model, Model2
from losses import L1L2Loss
import paths


def mk_img_mesh_transforms(image_transforms):

    def apply(img, mesh):
        return image_transforms(img), torch.from_numpy(mesh)
    return apply


def main():

    epochs = 10000
    batch_size = 32
    lr = 2.5
    device = 'cuda'
    epochs_split = 50

    backbone, n_backbone_features = pretrainedmodels.resnet18(), 512
    # backbone, n_backbone_features = pretrainedmodels.resnet34(), 512
    # backbone, n_backbone_features = pretrainedmodels.resnet50(), 1024
    # backbone, n_backbone_features = pretrainedmodels.resnet101(), 2048
    # backbone, n_backbone_features = pretrainedmodels.se_resnext50_32x4d(), 1024
    # backbone, n_backbone_features = pretrainedmodels.se_resnext101_32x4d(), 2048
    # backbone, n_backbone_features = pretrainedmodels.nasnetamobile(num_classes=1000), 1056

    # model = Model(backbone, n_backbone_features, 9591)
    model = Model2(backbone, n_backbone_features, 20, 9591)

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

    # criterion = nn.MSELoss()
    criterion = L1L2Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.fc_final.parameters(), lr=lr)
    # optimizer = optim.Adam(list(model.fc_final.parameters()) + list(model.fc1.parameters()), lr=lr)

    print("START ALL TRAINING")
    model.to(device)
    best_val = 1e10
    for epoch in range(1, epochs + 1):
        losses = []
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
        print(f"{epoch}/{epochs} loss = {mean_loss}, val_loss = {val_loss}")

        if epoch == 30:
            lr = 0.1
            print(f'new lr = {lr}')
            for param_group in optimizer.param_groups: param_group['lr'] = lr

        if epoch == 50:
            lr = 0.05
            print(f'new lr = {lr}')
            for param_group in optimizer.param_groups: param_group['lr'] = lr

        if epoch == 100:
            lr = 0.01
            print(f'new lr = {lr}')
            for param_group in optimizer.param_groups: param_group['lr'] = lr

        if epoch == 150:
            lr = 0.005
            print(f'new lr = {lr}')
            for param_group in optimizer.param_groups: param_group['lr'] = lr

        torch.save(model.state_dict(), "/work/checkpoints/current.pt")
        if val_loss < best_val:
            best_val = val_loss
            print("New best val loss", val_loss)
            torch.save(model.state_dict(), "/work/checkpoints/{:05d}_{}_{}.pt".format(
                epoch, mean_loss, val_loss))


if __name__ == '__main__':
    main()
