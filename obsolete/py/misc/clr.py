import numpy as np
import matplotlib.pyplot as plt


def getTriangularLr(stepsize, base_lr, max_lr):
    def f(iteration):
        cycle = np.floor(1.0 + float(iteration)/(2.0  * float(stepsize)))
        x = np.abs(float(iteration)/stepsize - 2.0 * cycle + 1.0)
        lr = base_lr + (max_lr - base_lr) * np.maximum(0.0, (1.0-x))
        return lr
    return f

# Demo of how the LR varies with iterations
num_iterations = 10000
stepsize = 1000
base_lr = 0.0001
max_lr = 0.001
lr_trend = list()

f = getTriangularLr(stepsize, base_lr, max_lr)
for iteration in range(num_iterations):
    lr = f(iteration)
    # Update your optimizer to use this learning rate in this iteration
    lr_trend.append(lr)

plt.plot(lr_trend)
plt.show()
