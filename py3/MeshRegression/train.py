import time
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

from sklearn.decomposition import PCA

from datasets import mk_kostet_dataset
from train_utils import run_validate
from models import *
from losses import L1L2Loss
import paths


def mk_pca_init(mesh_dataset, n_components):
    pca = PCA(n_components)
    data = []
    for _, mesh in mesh_dataset:
        data.append(mesh.flatten().numpy())

    data = np.array(data)
    pca.fit(data)
    pca.singular_values_
    print(pca.mean_.shape, pca.components_.shape)
    # return pca.mean_, np.diag(pca.singular_values_) @ pca.components_
    return pca.mean_, pca.components_


def mk_img_mesh_transforms(image_transforms):
    def apply(img, mesh):
        return image_transforms(img), torch.from_numpy(mesh)
    return apply


def main():
    epochs = 10000
    # batch_size = 16
    batch_size = 50
    lr = 0.00005
    device = 'cuda'

    # backbone, n_backbone_features = pretrainedmodels.resnet18(), 512
    backbone, n_backbone_features = pretrainedmodels.resnet34(), 512
    # backbone, n_backbone_features = pretrainedmodels.resnet50(), 1024
    # backbone, n_backbone_features = pretrainedmodels.resnet101(), 2048
    # backbone, n_backbone_features = pretrainedmodels.se_resnext50_32x4d(), 1024
    # backbone, n_backbone_features = pretrainedmodels.se_resnext101_32x4d(), 2048
    # backbone, n_backbone_features = pretrainedmodels.nasnetamobile(num_classes=1000), 1056

    # model = Model(backbone, n_backbone_features, 9591)
    # model = Model2(backbone, n_backbone_features, 160, 9591)
    model = FinNet(160, 9591)

    train_img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(backbone.mean, backbone.std),
    ])
    val_img_transforms = train_img_transforms
    train_transforms = mk_img_mesh_transforms(train_img_transforms)
    val_transforms = mk_img_mesh_transforms(val_img_transforms)

    # train_dataset = mk_kostet_dataset(list(range(0, 100)) + list(range(200, 299)), train_transforms)
    # val_dataset = mk_kostet_dataset(list(range(100, 200)), val_transforms)
    train_dataset = mk_kostet_dataset(list(range(0, 250)), train_transforms)
    val_dataset = mk_kostet_dataset(list(range(250, 299)), val_transforms)

    print(f"n train = {len(train_dataset)}, n val = {len(val_dataset)}")

    pca_mean, pca_std = mk_pca_init(train_dataset, 160)
    print(model.fc_final.bias.shape, model.fc_final.weight.shape)
    model.fc_final.bias.data = torch.FloatTensor(pca_mean)
    model.fc_final.weight.data = torch.FloatTensor(pca_std.T)
    model.fc_final.requires_grad = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    criterion = nn.MSELoss()
    # criterion = L1L2Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], 0.2)
    # optimizer = optim.Adam(model.fc_final.parameters(), lr=lr)
    # optimizer = optim.Adam(list(model.fc_final.parameters()) + list(model.fc1.parameters()), lr=lr)

    print("START ALL TRAINING")
    model.to(device)
    best_val = 1e10
    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        start = time.time()
        for inp, target in train_loader:
            inp, target = inp.to(device), target.to(device)
            output = model(inp)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
        model.eval()
        elapsed = time.time() - start

        scheduler.step(epoch)

        mean_loss = np.mean(losses)
        val_loss = run_validate(model, criterion, val_loader, device)
        val_l1l2 = run_validate(model, L1L2Loss(), val_loader, device)
        print(
            f"{epoch}/{epochs} "
            f"loss = {mean_loss}, "
            f"val_loss = {val_loss}, "
            f"val_l1l2 = {val_l1l2} "
            f"lr = {scheduler.get_lr()},"
            f"elpsd = {elapsed}")

        torch.save(model.state_dict(), "/work/checkpoints/current.pt")
        if val_loss < best_val:
            best_val = val_loss
            print("New best val loss", val_loss)
            torch.save(model.state_dict(), "/work/checkpoints/{:05d}_{}_{}.pt".format(
                epoch, mean_loss, val_loss))


if __name__ == '__main__':
    main()
