import random
import sys
import os
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import pretrainedmodels

from datasets import mk_kostet_dataset
import paths


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

    batch_size = 32

    backbone = pretrainedmodels.resnet18()
    model = Model(backbone, 512 * 4 * 3, 9591 * 3)

    train_transforms = None
    val_transforms = None

    train_dataset = mk_kostet_dataset(list(range(0, 100)) + list(range(200, 299)), train_transforms)
    val_dataset = mk_kostet_dataset(list(range(100, 200)), val_transforms)

    print(f"n train = {len(train_dataset)}, n val = {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


if __name__ == '__main__':
    main()
