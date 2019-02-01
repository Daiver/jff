from collections import OrderedDict
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

import torch_fuze
import mlflow

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


def main(
        epochs,
        batch_size,
        lr,
        n_pca_components,
        device
):
    # backbone, n_backbone_features = pretrainedmodels.resnet18(), 512
    backbone, n_backbone_features = pretrainedmodels.resnet34(), 512
    # backbone, n_backbone_features = pretrainedmodels.resnet50(), 1024
    # backbone, n_backbone_features = pretrainedmodels.resnet101(), 2048
    # backbone, n_backbone_features = pretrainedmodels.se_resnext50_32x4d(), 1024
    # backbone, n_backbone_features = pretrainedmodels.se_resnext101_32x4d(), 2048
    # backbone, n_backbone_features = pretrainedmodels.nasnetamobile(num_classes=1000), 1056

    # model = Model(backbone, n_backbone_features, 9591)
    # model = Model2(backbone, n_backbone_features, 160, 9591)
    model = FinNet(n_pca_components, 9591)

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

    pca_mean, pca_std = mk_pca_init(train_dataset, n_pca_components)
    print(model.fc_final.bias.shape, model.fc_final.weight.shape)
    model.fc_final.bias.data = torch.FloatTensor(pca_mean)
    model.fc_final.weight.data = torch.FloatTensor(pca_std.T)
    model.fc_final.requires_grad = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = L1L2Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150, 200], 0.2)
    # optimizer = optim.Adam(model.fc_final.parameters(), lr=lr)
    # optimizer = optim.Adam(list(model.fc_final.parameters()) + list(model.fc1.parameters()), lr=lr)

    print("START ALL TRAINING")

    metrics = OrderedDict([
        ("loss", criterion),
        ("l1", nn.L1Loss()),
        ("l2", nn.MSELoss()),
        ("l1_2", L1L2Loss())
    ])
    callbacks = [
        torch_fuze.callbacks.ProgressCallback(),
        torch_fuze.callbacks.BestModelSaverCallback(
            model, "checkpoints/best.pt", metric_name="loss", lower_is_better=True),
        torch_fuze.callbacks.TensorBoardXCallback("logs", remove_old_logs=True),
        torch_fuze.callbacks.MLFlowCallback(
            lowest_metrics_to_track={"valid_loss", "valid_l1_2", "train_loss"},
            files_to_save_at_every_batch={"checkpoints/best.pt"})
    ]
    trainer = torch_fuze.SupervisedTrainer(model, criterion, device)
    trainer.run(
        train_loader, val_loader, optimizer, scheduler=scheduler, n_epochs=epochs, callbacks=callbacks, metrics=metrics
    )


if __name__ == '__main__':
    with mlflow.start_run(run_name="L2 loss"):
        params = {
            "epochs": 100,
            "batch_size": 50,
            "lr": 0.001,
            "n_pca_components": 160,
        }
        for k, v in params.items():
            mlflow.log_param(k, v)
        main(
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            n_pca_components=params["n_pca_components"],
            device="cuda"
        )
    with mlflow.start_run(run_name="L2 loss"):
        params = {
            "epochs": 100,
            "batch_size": 50,
            "lr": 0.0005,
            "n_pca_components": 160,
        }
        for k, v in params.items():
            mlflow.log_param(k, v)
        main(
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            n_pca_components=params["n_pca_components"],
            device="cuda"
        )
        with mlflow.start_run(run_name="L2 loss"):
            params = {
                "epochs": 100,
                "batch_size": 50,
                "lr": 0.0002,
                "n_pca_components": 160,
            }
            for k, v in params.items():
                mlflow.log_param(k, v)
            main(
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                lr=params["lr"],
                n_pca_components=params["n_pca_components"],
                device="cuda"
            )
        with mlflow.start_run(run_name="L2 loss"):
            params = {
                "epochs": 100,
                "batch_size": 50,
                "lr": 0.0001,
                "n_pca_components": 160,
            }
            for k, v in params.items():
                mlflow.log_param(k, v)
            main(
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                lr=params["lr"],
                n_pca_components=params["n_pca_components"],
                device="cuda"
            )
