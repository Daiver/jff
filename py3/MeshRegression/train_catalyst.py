import catalyst
from catalyst.dl.runner import SupervisedModelRunner
import collections

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

from train import mk_img_mesh_transforms


def main():
    n_epochs = 10000
    batch_size = 16
    lr = 0.02
    device = 'cuda'

    backbone, n_backbone_features = pretrainedmodels.resnet18(), 512
    # backbone, n_backbone_features = pretrainedmodels.resnet34(), 512
    # backbone, n_backbone_features = pretrainedmodels.resnet50(), 1024
    # backbone, n_backbone_features = pretrainedmodels.resnet101(), 2048
    # backbone, n_backbone_features = pretrainedmodels.se_resnext50_32x4d(), 1024
    # backbone, n_backbone_features = pretrainedmodels.se_resnext101_32x4d(), 2048
    # backbone, n_backbone_features = pretrainedmodels.nasnetamobile(num_classes=1000), 1056

    # model = Model(backbone, n_backbone_features, 9591)
    model = Model2(backbone, n_backbone_features, 100, 9591)

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

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = val_loader

    criterion = nn.MSELoss()
    # criterion = L1L2Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 70, 200], 0.1)

    from catalyst.dl.callbacks import (
        LossCallback,
        Logger, TensorboardLogger,
        OptimizerCallback, SchedulerCallback, CheckpointCallback,
        PrecisionCallback, OneCycleLR)

    logdir = "logs/"
    callbacks = collections.OrderedDict()
    callbacks["loss"] = LossCallback()
    callbacks["optimizer"] = OptimizerCallback()
    callbacks["scheduler"] = OneCycleLR(
        cycle_len=n_epochs,
        div=3, cut_div=4, momentum_range=(0.95, 0.85))
    callbacks["saver"] = CheckpointCallback()
    callbacks["logger"] = Logger()
    callbacks["tflogger"] = TensorboardLogger()

    runner = SupervisedModelRunner(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler)
    runner.train(
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        epochs=n_epochs, verbose=True)


if __name__ == '__main__':
    main()
