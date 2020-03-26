import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams
# rcParams['figure.figsize'] = 10, 8
# rcParams['figure.dpi'] = 300

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler


class RealNVP(nn.Module):
    def __init__(self, nets, nett, masks, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        return -(torch.einsum('bs,bs->b', z, z)) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z.cuda())
        return x


def main():
    net_s = lambda: nn.Sequential(
        nn.Linear(2, 256), nn.LeakyReLU(),
        nn.Linear(256, 256), nn.LeakyReLU(),
        nn.Linear(256, 2), nn.Tanh()
    ).cuda()
    net_t = lambda: nn.Sequential(
        nn.Linear(2, 256), nn.LeakyReLU(),
        nn.Linear(256, 256), nn.LeakyReLU(),
        nn.Linear(256, 2)
    ).cuda()
    masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32)).cuda()
    prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    flow = RealNVP(net_s, net_t, masks, prior).cuda()

    optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=2e-3)
    # batch_size = 100
    batch_size = 2048
    for t in range(5001):
        noisy_moons = datasets.make_moons(n_samples=batch_size, noise=.05)[0].astype(np.float32)
        loss = -flow.log_prob(torch.from_numpy(noisy_moons).cuda()).mean()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if t % 500 == 0:
            print('iter %s:' % t, 'loss = %.3f' % loss)

            noisy_moons = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
            z = flow.f(torch.from_numpy(noisy_moons).cuda())[0].cpu().detach().numpy()
            plt.subplot(221)
            plt.scatter(z[:, 0], z[:, 1])
            plt.title(r'$z = f(X)$')

            z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)
            plt.subplot(222)
            plt.scatter(z[:, 0], z[:, 1])
            plt.title(r'$z \sim p(z)$')

            plt.subplot(223)
            x = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
            plt.scatter(x[:, 0], x[:, 1], c='r')
            plt.title(r'$X \sim p(X)$')

            plt.subplot(224)
            x = flow.sample(1000).cpu().detach().numpy()
            plt.scatter(x[:, 0, 0], x[:, 0, 1], c='r')
            plt.title(r'$X = g(z)$')
            plt.show()


if __name__ == '__main__':
    main()
