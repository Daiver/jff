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

from datasets import mk_kostet_dataset, mk_synth_dataset_test, mk_synth_dataset_train

from models import *
from utils import replace_obj_vertices

from train import mk_img_mesh_transforms


def main():
    # backbone = pretrainedmodels.resnet18()
    # backbone = pretrainedmodels.resnet34()
    # model = Model(backbone, 512, 9591)
    # backbone, n_backbone_features = pretrainedmodels.resnet18(), 512
    backbone, n_backbone_features = pretrainedmodels.resnet34(), 512

    # n_vertices = 9591
    n_vertices = 5958
    # path_to_default_obj = "/work/R3DS/Data/MeshRegression/KostetSmoothCutted/Object000.obj"
    path_to_default_obj = "/home/daiver/test_geoms/Object000.obj"

    # model = Model(backbone, n_backbone_features, 9591)
    # model = Model2(backbone, n_backbone_features, 100, 9591)
    model = FinNet(400, n_vertices)

    path_to_model = "checkpoints/best_2019-02-08_10:26:08.pt"
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    train_img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(backbone.mean, backbone.std),
    ])
    transforms = mk_img_mesh_transforms(train_img_transforms)

    # train_dataset = mk_kostet_dataset(list(range(0, 299)), train_transforms)
    dataset = mk_synth_dataset_test(transform=transforms)
    # dataset = mk_synth_dataset_train(transform=transforms)

    print(f"n train = {len(dataset)}")

    # path_to_model = "checkpoints/999_0.07310015199537988.pt"
    # path_to_model = "checkpoints2/620_0.11420603259766532.pt"

    batch_size = 128
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    obj_ind = 0
    print("Start activation")
    for img_batch, _ in train_loader:
        vertices = model(img_batch)
        for i in range(vertices.size(0)):
            replace_obj_vertices(path_to_default_obj, vertices[i], "/home/daiver/res/Object{:03d}.obj".format(obj_ind))
            obj_ind += 1
            # img = img_batch[0].detach().cpu().numpy()
            # img = np.moveaxis(img, 0, 2)
            # img *= 0.229, 0.224, 0.22
            # img += 0.5, 0.5, 0.5
            # img *= 255
            # img = img.astype(np.uint8)
            # print(img.shape)
            # cv2.imshow('', img)
            # cv2.waitKey()


if __name__ == '__main__':
    main()
