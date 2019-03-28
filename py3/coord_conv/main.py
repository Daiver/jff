import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from collections import OrderedDict
import random
import cv2
import numpy as np
import torch
import torch_fuze

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from coord_conv import CoordConv2d


canvas_size = (64, 64)


def mk_point_sample(coord_limits_row=(0, canvas_size[0] - 1), coord_limits_col=(0, canvas_size[1] - 1)):
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    coord = (
        random.randint(coord_limits_row[0], coord_limits_row[1]),
        random.randint(coord_limits_col[0], coord_limits_col[1]))
    canvas[coord[0], coord[1]] = (0, 255, 0)
    return canvas, coord


def mk_uniform_points_dataset(n_train_samples, n_test_samples):
    return [mk_point_sample() for _ in range(n_train_samples)], [mk_point_sample() for _ in range(n_test_samples)]


def mk_quad_points_dataset(n_train_samples, n_test_samples):
    s1 = [
        mk_point_sample(coord_limits_row=(0, 31), coord_limits_col=(0, 31))
        for _ in range(n_train_samples // 3)
    ]
    s2 = [
        mk_point_sample(coord_limits_row=(32, 63), coord_limits_col=(0, 31))
        for _ in range(n_train_samples // 3)
    ]
    s3 = [
        mk_point_sample(coord_limits_row=(0, 31), coord_limits_col=(32, 63))
        for _ in range(n_train_samples // 3)
    ]
    s4 = [
        mk_point_sample(coord_limits_row=(32, 63), coord_limits_col=(32, 63))
        for _ in range(n_test_samples)
    ]
    return s1 + s2 + s3, s4


def main_view_set():
    torch_fuze.utils.manual_seed(42)
    train_set, test_set = mk_quad_points_dataset(20, 20)
    print("TRAIN SET:")
    for img, target in train_set:
        print(target)
        cv2.imshow('', cv2.pyrUp(cv2.pyrUp(img)))
        cv2.waitKey()
    print("TEST SET:")
    for img, target in test_set:
        print(target)
        cv2.imshow('', cv2.pyrUp(cv2.pyrUp(img)))
        cv2.waitKey()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),  # 64 -> 32
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 4 -> 2
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )

        self.fc = nn.Linear(2 * 2 * 16, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 2 * 2 * 16)
        x = self.fc(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            CoordConv2d(3, 16, 5, stride=2, padding=2),  # 64 -> 32
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            CoordConv2d(16, 16, 3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            CoordConv2d(16, 16, 3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            CoordConv2d(16, 16, 3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            CoordConv2d(16, 16, 3, stride=2, padding=1),  # 4 -> 2
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )

        self.fc = nn.Linear(2 * 2 * 16, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 2 * 2 * 16)
        x = self.fc(x)
        return x


class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 16)
        x = self.fc(x)
        return x


class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            CoordConv2d(3, 8, 1, stride=1, padding=0),  # 64 -> 32
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 8, 1, stride=1, padding=0),  # 64 -> 32
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 8, 1, stride=1, padding=0),  # 64 -> 32
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),  # 64 -> 32
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 2, 3, stride=1, padding=1),  # 32 -> 16
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 2)
        # x = self.fc(x)
        return x


def main_train():
    print(f"Torch version: {torch.__version__}, CUDA: {torch.version.cuda}, Fuze version: {torch_fuze.__version__}")

    torch_fuze.utils.manual_seed(42)
    n_train_samples = 2000
    n_test_samples = 1000
    train_set, test_set = mk_uniform_points_dataset(n_train_samples, n_test_samples)
    # train_set, test_set = mk_quad_points_dataset(n_train_samples, n_test_samples)

    use_ordinary_model = True
    use_ordinary_model = False

    # lr = 0.01
    # batch_size = 32
    # batch_size = 64
    batch_size = 128
    # batch_size = 512
    # device = "cpu"
    device = "cuda:0"

    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU name: {torch.cuda.get_device_name(int(device.split(':')[-1]))}")

    inp_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    out_trans = torch.FloatTensor

    train_loader = DataLoader(torch_fuze.data.InputOutputTransformsWrapper(train_set, inp_trans, out_trans),
                              batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    test_loader = DataLoader(torch_fuze.data.InputOutputTransformsWrapper(test_set, inp_trans, out_trans),
                             batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)

    if use_ordinary_model:
        model = Net()
    else:
        model = Net2()
    model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=2e-4)
    scheduler = None
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 600], gamma=0.2)

    # best_lr = torch_fuze.utils.find_lr_supervised(
    #     model, criterion, optimizer, train_loader, 0.0001, 1, device=device, n_iterations=100)[0]
    # best_lr = 0.9120108393559097
    # print("best lr", best_lr)
    # torch_fuze.utils.manual_seed(42)
    # torch_fuze.utils.set_lr(optimizer, best_lr)
    # # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 100], gamma=0.1)
    # cycle_len = int(200 * len(train_loader))
    # print("cycle_len", cycle_len)
    # scheduler = torch_fuze.lr_scheduler.OneCycleLR(
    #     optimizer, 1e-6, best_lr * 0.1, 1e-6, cycle_len)

    metrics = OrderedDict([
        ("loss", criterion),
        # ("acc", torch_fuze.metrics.Accuracy())
    ])
    log_dir = "conv" if use_ordinary_model else "coord"
    callbacks = [
        torch_fuze.callbacks.ProgressCallback(),
        torch_fuze.callbacks.TensorBoardXCallback(log_dir=f"logs/{log_dir}/", remove_old_logs=True)
    ]
    trainer = torch_fuze.SupervisedTrainer(model, criterion, device)
    trainer.run(
        train_loader, test_loader, optimizer,
        scheduler=scheduler, n_epochs=1000, callbacks=callbacks, metrics=metrics,
        scheduler_after_each_batch=False)

    if use_ordinary_model:
        print("Conv training finished")
    else:
        print("CoordConv training finished")


if __name__ == '__main__':
    # main_view_set()
    main_train()
