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
import paths


def mk_img_mesh_transforms(image_transforms):
    to_tensor = torchvision.transforms.ToTensor()
    
    def apply(img, mesh):
        return image_transforms(img), to_tensor(mesh)
    return apply


class Model(nn.Module):
    def __init__(self, backbone, n_final_features, n_outputs):
        super().__init__()
        self.add_module("backbone", backbone)
        self.backbone = backbone
        self.n_final_features = n_final_features
        self.fc_final = nn.Linear(n_final_features, n_outputs)

    def forward(self, x):
        x = self.backbone.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.n_final_features)
        x = self.fc_final(x)

        return x


def main():

    epochs = 100
    batch_size = 32
    lr = 0.002

    backbone = pretrainedmodels.resnet18()
    model = Model(backbone, 512 * 4 * 3, 9591 * 3)

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

    losses = []
    for epoch in range(epochs):
        model.train()
        for inp, target in train_loader:
            output = model(inp)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss)
        mean_loss = np.mean(losses)
        print(f"{epoch}/{epochs} loss = {mean_loss}")
        torch.save(model, f"checkpoints/{epoch}_{mean_loss}.pt")


if __name__ == '__main__':
    main()
