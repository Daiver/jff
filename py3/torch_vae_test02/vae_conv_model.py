import torch
from torch import nn, optim
from torch.nn import functional as F


class VAEConv(nn.Module):
    def __init__(self, latent_size: int):
        self.latent_size = latent_size
        self.hidden_size = 64

        # activation = nn.ELU
        # activation = lambda: nn.LeakyReLU(negative_slope=1e-2)
        activation = nn.ReLU
        super().__init__()
        self.encoder_base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2, stride=2),  # -> 14x14
            nn.BatchNorm2d(8),
            activation(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, stride=2),  # -> 7x7
            nn.BatchNorm2d(16),
            activation(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2),  # -> 4x4
            nn.BatchNorm2d(32),
            activation(),
            nn.Conv2d(in_channels=32, out_channels=self.hidden_size, kernel_size=4, padding=0, stride=1),  # -> 1x1
            nn.BatchNorm2d(self.hidden_size),
            activation(),
        )

        self.fc21 = nn.Linear(self.hidden_size, latent_size)
        self.fc22 = nn.Linear(self.hidden_size, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 400),
            nn.BatchNorm1d(400),
            activation(),
            nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            activation(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(-1, 1, 28, 28)
        h1 = self.encoder_base(x)
        # print(h1.shape)
        h1 = h1.view(-1, self.hidden_size)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        # print(decoded.shape, mu.shape, x.shape)
        return decoded, mu, logvar
