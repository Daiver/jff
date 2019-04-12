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


canvas_size = (32, 32)


def mk_point_sample():
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    coord = (random.randint(0, canvas_size[0] - 1), random.randint(0, canvas_size[1] - 1))
    canvas[coord[0], coord[1]] = (0, 255, 0)
    return canvas, coord


def mk_points_dataset(n_samples):
    return [mk_point_sample() for _ in range(n_samples)]


def main():
    torch_fuze.utils.manual_seed(42)
    n_samples = 200
    dataset = mk_points_dataset(n_samples)
    for img, target in dataset:
        print(target)
        cv2.imshow('', cv2.pyrUp(cv2.pyrUp(img)))
        cv2.waitKey()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),  # 32 -> 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),  # 16 -> 8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 4 -> 2
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.fc = nn.Linear(2 * 2 * 32, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 2 * 2 * 32)
        x = self.fc(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            CoordConv2d(3, 16, 5, stride=2, padding=2),  # 32 -> 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            CoordConv2d(16, 32, 5, stride=2, padding=2),  # 16 -> 8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            CoordConv2d(32, 32, 3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            CoordConv2d(32, 32, 3, stride=2, padding=1),  # 4 -> 2
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.fc = nn.Linear(2 * 2 * 32, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 2 * 2 * 32)
        x = self.fc(x)
        return x


def main_train():
    print(f"Torch version: {torch.__version__}, CUDA: {torch.version.cuda}, Fuze version: {torch_fuze.__version__}")

    torch_fuze.utils.manual_seed(42)
    n_train_samples = 1000
    n_test_samples = 200
    train_set = mk_points_dataset(n_train_samples)
    test_set = mk_points_dataset(n_test_samples)

    # lr = 0.01
    # batch_size = 32
    # batch_size = 64
    batch_size = 256
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

    model = Net()
    # model = Net2()
    model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.2)

    metrics = OrderedDict([
        ("loss", criterion),
        # ("acc", torch_fuze.metrics.Accuracy())
    ])
    callbacks = [
        torch_fuze.callbacks.ProgressCallback(),
    ]
    trainer = torch_fuze.SupervisedTrainer(model, criterion, device)
    trainer.run(
        train_loader, test_loader, optimizer, scheduler=scheduler, n_epochs=200, callbacks=callbacks, metrics=metrics)


if __name__ == '__main__':
    # main()
    main_train()
