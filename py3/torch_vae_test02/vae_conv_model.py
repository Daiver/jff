import torch
from torch import nn, optim
from torch.nn import functional as F


class VAEConv(nn.Module):
    def __init__(self, latent_size: int):
        self.latent_size = latent_size
        self.n_last_filters = 32

        # activation = nn.ELU
        activation = lambda num_parameters: nn.PReLU(num_parameters=num_parameters)
        # activation = lambda: nn.LeakyReLU(negative_slope=1e-2)
        # activation = nn.ReLU
        super().__init__()
        self.encoder_base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),  # -> 28x28
            nn.BatchNorm2d(8),
            activation(8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=2),  # -> 14x14
            nn.BatchNorm2d(8),
            activation(8),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, stride=2),  # -> 7x7
            nn.BatchNorm2d(16),
            activation(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2),  # -> 4x4
            nn.BatchNorm2d(32),
            activation(32),
            nn.Conv2d(in_channels=32, out_channels=self.n_last_filters, kernel_size=4, padding=0, stride=1),  # -> 1x1
            nn.BatchNorm2d(self.n_last_filters),
            activation(self.n_last_filters),
        )

        self.fc21 = nn.Linear(self.n_last_filters, latent_size)
        self.fc22 = nn.Linear(self.n_last_filters, latent_size)

        self.decoder_fc = nn.Linear(latent_size, 3 * 3 * self.n_last_filters)
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_last_filters, out_channels=32,
                kernel_size=3, padding=2, stride=1),  # -> 5x5
            nn.BatchNorm2d(32),
            activation(32),

            nn.Conv2d(
                in_channels=self.n_last_filters, out_channels=32,
                kernel_size=3, padding=2, stride=1),  # -> 7x7
            nn.BatchNorm2d(32),
            activation(32),

            nn.UpsamplingBilinear2d(scale_factor=2),  # -> 14x14
            nn.Conv2d(
                in_channels=32, out_channels=16,
                kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(16),
            activation(16),
            nn.UpsamplingBilinear2d(scale_factor=2),  # -> 28x28
            nn.Conv2d(
                in_channels=16, out_channels=8,
                kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(8),
            activation(8),
            nn.Conv2d(
                in_channels=8, out_channels=8,
                kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(8),
            activation(8),
            nn.Conv2d(
                in_channels=8, out_channels=1,
                kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(-1, 1, 28, 28)
        h1 = self.encoder_base(x)
        # print(h1.shape)
        h1 = h1.view(-1, self.n_last_filters)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(-1, self.n_last_filters, 3, 3)
        res = self.decoder(z)
        # print(res.shape)
        return res.view(-1, 28 * 28)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        # print(decoded.shape, mu.shape, x.shape)
        return decoded, mu, logvar
