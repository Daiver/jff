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

from datasets import mk_kostet_dataset
from train_utils import run_validate
from models import Model
from utils import replace_obj_vertices

from train import mk_img_mesh_transforms


def main():
    backbone = pretrainedmodels.resnet18()
    model = Model(backbone, 512, 9591)

    train_img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(backbone.mean, backbone.std),
    ])
    val_img_transforms = train_img_transforms
    train_transforms = mk_img_mesh_transforms(train_img_transforms)
    val_transforms = mk_img_mesh_transforms(val_img_transforms)

    train_dataset = mk_kostet_dataset(list(range(0, 299)), train_transforms)
    val_dataset = mk_kostet_dataset(list(range(100, 200)), val_transforms)

    print(f"n train = {len(train_dataset)}, n val = {len(val_dataset)}")

    # path_to_model = "checkpoints/999_0.07310015199537988.pt"
    path_to_model = "checkpoints2/620_0.11420603259766532.pt"
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    path_to_default_obj = "/work/R3DS/Data/MeshRegression/KostetSmoothCutted/Object000.obj"
    obj_ind = 0
    for img, _ in train_loader:
        vertices = model(img)
        for i in range(vertices.size(0)):
            replace_obj_vertices(path_to_default_obj, vertices[i], "/home/daiver/res/Object{:03d}.obj".format(obj_ind))
            obj_ind += 1


if __name__ == '__main__':
    main()
