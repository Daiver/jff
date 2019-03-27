from collections import OrderedDict
import torch
import torch_fuze

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def mk_dataset()


def main():
    print(f"Torch version: {torch.__version__}, CUDA: {torch.version.cuda}, Fuze version: {torch_fuze.__version__}")

    # lr = 0.01
    # batch_size = 32
    batch_size = 64
    # batch_size = 128
    # device = "cpu"
    device = "cuda:0"

    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU name: {torch.cuda.get_device_name(int(device.split(':')[-1]))}")

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if not exist, download mnist dataset
    train_set = CIFAR10(root="data/", train=True, transform=trans, download=True)
    test_set = CIFAR10(root="data/", train=False, transform=trans, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)

    # model = Net()
    model = Net2()
    model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.3)

    metrics = OrderedDict([
        ("loss", criterion),
        ("acc", torch_fuze.metrics.Accuracy())
    ])
    callbacks = [
        torch_fuze.callbacks.ProgressCallback(),
    ]
    trainer = torch_fuze.SupervisedTrainer(model, criterion, device)
    trainer.run(
        train_loader, test_loader, optimizer, scheduler=scheduler, n_epochs=200, callbacks=callbacks, metrics=metrics)


if __name__ == '__main__':
    main()
