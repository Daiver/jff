import matplotlib.pyplot as plt
import numpy as np
import math

import torch.nn as nn
import torch.optim as optim

import lr_schedulers as custom_schedulers


if __name__ == '__main__':
    n_steps = 100

    model = nn.Sequential(nn.Linear(1, 1))
    lr = 1
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scheduler = custom_schedulers.GeomRampUpLinearDecayLR(
        optimizer,
        n_iters=n_steps, n_rampup_iters=5, n_iters_before_decay=30,
        rampup_gamma=2.0, middle_coeff=0.0)

    xs, ys = [], []
    for i in range(n_steps):
        xs.append(i)
        scheduler.step()
        ys.append(scheduler.get_lr()[0])

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    plt.plot(xs, ys, 'ro')
    plt.show()
