import numpy as np

import cv2
import np_draw_tools

import torch
import torch.utils.data
import datetime
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from vae_fc_model import VAEFC
from vae_conv_model_small import VAEConv


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def run_training_stage(model, optimizer, train_loader, device, log_interval, epoch):
    model.train()
    train_loss = 0

    start_time = datetime.datetime.now()
    prefix = 'vanila'

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        # print(labels)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    torch.save(model, f'checkpoints/{prefix}_{str(start_time)}_{epoch}.pt'.replace(":", "_"))


def run_testing_stage(model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def rotation_matrix2d(theta):
    return np.array((
        (np.cos(theta), -np.sin(theta)),
        (np.sin(theta),  np.cos(theta))
    ))


def draw_figure(angle):
    canvas_size = (28, 28)
    canvas_center = np.array([canvas_size[0] / 2, canvas_size[1] / 2])
    canvas = np.zeros(canvas_size + (1,), dtype=np.uint8)
    lines = np.array([
        (5, 14),
        (14, 14),
        (14, 5),
        (14, 14),
    ], dtype=np.float32)
    rot_mat = rotation_matrix2d(angle)
    lines = lines - canvas_center
    lines = lines @ rot_mat.T
    lines = lines + canvas_center

    for i in range(len(lines) // 2):
        x1, y1 = lines[2 * i + 0]
        x2, y2 = lines[2 * i + 1]
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.line(canvas, pt1, pt2, 255, thickness=3)
    return canvas


def main():
    train_samples = [draw_figure(angle) for angle in np.linspace(0, 2*np.pi, 32 + 1)][:-1]
    grid = np_draw_tools.make_grid(train_samples)
    cv2.imshow("", grid)
    cv2.waitKey(1000)

    train_dataset_torch = [TF.to_tensor(x) for x in train_samples]
    test_dataset_torch = [TF.to_tensor(x) for x in train_samples]

    torch.manual_seed(42)

    epochs = 2000
    # batch_size = 32
    batch_size = 32
    device = "cuda:0"
    latent_size = 1
    log_interval = 1

    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset_torch, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset_torch, batch_size=batch_size, shuffle=True, **kwargs)

    # model = VAEFC(hidden_size=10, latent_size=latent_size).to(device)
    model = VAEConv(latent_size=latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    sample = torch.linspace(-1, 1, 256).view(-1, 1).to(device)
    for epoch in range(1, epochs + 1):
        run_training_stage(model, optimizer, train_loader, device, log_interval, epoch)
        run_testing_stage(model, test_loader, device, epoch)
        with torch.no_grad():
            sample_gen = model.decode(sample).cpu()
            save_image(sample_gen.view(-1, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')


if __name__ == "__main__":
    main()
