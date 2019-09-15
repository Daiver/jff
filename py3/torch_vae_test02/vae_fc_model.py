import torch
from torch import nn, optim
from torch.nn import functional as F


class VAEFC(nn.Module):
    def __init__(self, hidden_size: int, latent_size: int):
        # activation = nn.ReLU()
        activation = nn.ELU
        super(VAEFC, self).__init__()
        self.encoder_base = nn.Sequential(
            nn.Linear(784, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
        )
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h1 = self.encoder_base(x)
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
        return self.decode(z), mu, logvar
