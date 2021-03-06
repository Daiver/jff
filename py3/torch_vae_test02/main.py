import torch
import torch.utils.data
import datetime
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from vae_fc_model import VAEFC
from vae_conv_model import VAEConv


<<<<<<< HEAD
torch.manual_seed(42)

epochs = 50
# batch_size = 32
batch_size = 64
device = "cuda:0"
latent_size = 16
hidden_size = 400
log_interval = 100


kwargs = {'num_workers': 8, 'pin_memory': True} 
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


# model = VAEFC(hidden_size=400, latent_size=latent_size).to(device)
model = VAEConv(latent_size=latent_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[15, 25])


=======
>>>>>>> 1eedf44801ae14674847d90fb7ed2d8673b574a7
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

    for batch_idx, (data, labels) in enumerate(train_loader):
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
        for i, (data, labels) in enumerate(test_loader):
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


<<<<<<< HEAD
if __name__ == "__main__":
    sample_lattent = torch.randn(256, latent_size).to(device)
    for epoch in range(1, epochs + 1):
        run_training_stage(epoch)
        lr_scheduler.step()
        run_testing_stage(epoch)
=======
def main():

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data', train=False, transform=transforms.ToTensor())

    torch.manual_seed(42)

    epochs = 20
    # batch_size = 32
    batch_size = 512
    device = "cuda:0"
    latent_size = 16
    hidden_size = 400
    log_interval = 1

    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # model = VAEFC(hidden_size=400, latent_size=latent_size).to(device)
    model = VAEConv(latent_size=latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        run_training_stage(model, optimizer, train_loader, device, log_interval, epoch)
        run_testing_stage(model, test_loader, device, epoch)
>>>>>>> 1eedf44801ae14674847d90fb7ed2d8673b574a7
        with torch.no_grad():
            sample = model.decode(sample_lattent).cpu()
            save_image(sample.view(256, 1, 28, 28),
<<<<<<< HEAD
                       'results/sample_' + str(epoch) + '.png', nrow=32)
=======
                       'results/sample_' + str(epoch) + '.png')


if __name__ == "__main__":
    main()
>>>>>>> 1eedf44801ae14674847d90fb7ed2d8673b574a7
