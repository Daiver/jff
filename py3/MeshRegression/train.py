from collections import OrderedDict
import datetime
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

from datasets import mk_kostet_dataset, mk_synth_dataset_test, mk_synth_dataset_train

from models import *
from losses import L1L2Loss
import paths
import lr_schedulers as custom_schedulers


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


def train(
        params,
        run_start_time
):
    print(params)
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    n_pca_components = params["n_pca_components"]
    dropout = params["dropout"]
    device = params["device"]

    readable_start_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(run_start_time / 1000))

    best_checkpoint_name = f"checkpoints/best_{readable_start_time}.pt"

    # n_vertices = 9591
    n_vertices = 5958

    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU name: {torch.cuda.get_device_name(int(device.split(':')[-1]))}")

    # backbone, n_backbone_features = pretrainedmodels.resnet18(), 512
    backbone, n_backbone_features = pretrainedmodels.resnet34(), 512

    model = FinNet(n_pca_components, n_vertices, dropout_val=dropout)

    train_img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(backbone.mean, backbone.std),
    ])
    val_img_transforms = train_img_transforms
    train_transforms = mk_img_mesh_transforms(train_img_transforms)
    val_transforms = mk_img_mesh_transforms(val_img_transforms)

    # train_dataset = mk_kostet_dataset(list(range(0, 100)) + list(range(200, 299)), train_transforms)
    # val_dataset = mk_kostet_dataset(list(range(100, 200)), val_transforms)

    train_dataset = mk_synth_dataset_train(transform=train_transforms)
    val_dataset = mk_synth_dataset_test(transform=val_transforms)

    print(f"n train = {len(train_dataset)}, n val = {len(val_dataset)}")

    pca_mean, pca_std = mk_pca_init(train_dataset, n_pca_components)
    print(model.fc_final.bias.shape, model.fc_final.weight.shape)
    model.fc_final.bias.data = torch.FloatTensor(pca_mean)
    # model.fc_final.weight.data = torch.FloatTensor(pca_std.T)
    # model.fc_final.requires_grad = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = L1L2Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150, 200], 0.2)
    scheduler = custom_schedulers.GeomRampUpLinearDecayLR(optimizer, epochs, 10, 20, rampup_gamma=1.3, middle_coeff=0.5)
    # scheduler = None

    print("START ALL TRAINING")

    metrics = OrderedDict([
        ("loss", criterion),
        # ("l1", nn.L1Loss()),
        # ("l2", nn.MSELoss()),
        ("l1_2", L1L2Loss())
    ])
    callbacks = [
        torch_fuze.callbacks.ProgressCallback(),
        torch_fuze.callbacks.BestModelSaverCallback(
            model, best_checkpoint_name, metric_name="l1_2", lower_is_better=True),
        torch_fuze.callbacks.TensorBoardXCallback(
            f"logs/{readable_start_time}", remove_old_logs=True),
        torch_fuze.callbacks.MLFlowCallback(
            lowest_metrics_to_track={"valid_loss", "valid_l1_2", "train_loss"},
            files_to_save_at_every_batch={best_checkpoint_name})
    ]
    trainer = torch_fuze.SupervisedTrainer(model, criterion, device)
    trainer.run(
        train_loader, val_loader, optimizer, scheduler=scheduler, n_epochs=epochs, callbacks=callbacks, metrics=metrics
    )
    print(f"Finished. device: {device}")


if __name__ == '__main__':

    with mlflow.start_run(run_name="FS3, Eyes") as run:
        params = {
            "epochs": 100,
            "batch_size": 32,
            "lr": 0.00002,
            "n_pca_components": 400,
            "dropout": 0.1,
            "device": "cuda:1",
        }
        for k, v in params.items():
            mlflow.log_param(k, v)
        train(params=params, run_start_time=run.info.start_time)

    # with mlflow.start_run(run_name="") as run:
    #     params = {
    #         "epochs": 100,
    #         "batch_size": 32,
    #         "lr": 0.00005,
    #         "n_pca_components": 400,
    #         "dropout": 0.1,
    #         "device": "cuda:1",
    #     }
    #     for k, v in params.items():
    #         mlflow.log_param(k, v)
    #     train(params=params, run_start_time=run.info.start_time)
