import numpy as np
import torch
import torch.nn.functional as F
import cv2


def sigmoid(inp):
    return F.sigmoid(1000*inp)


def dist_loss(inp, dist_field):
    inp = sigmoid(inp)
    return (inp * dist_field).sum()


def len_loss(inp):
    inp = sigmoid(inp)
    print(inp.sum())
    # return (inp.sum() - 1.0) ** 2
    return (inp[2, 2] - 1.0) ** 2


dist_field = torch.Tensor([
    [2, 2, 2, 2, 2],
    [2, 1, 1, 1, 2],
    [2, 1, 0, 1, 2],
    [2, 1, 1, 1, 2],
    [2, 2, 2, 2, 2]
])

vars = torch.zeros(5, 5, requires_grad=True)

print(dist_field)
print(vars)

step_len = 0.01
for iter_ind in range(100):
    loss = dist_loss(vars, dist_field) + len_loss(vars)
    # loss = len_loss(vars)
    loss.backward()
    # vars = vars - step_len * vars.grad
    vars.data.add_(-step_len * vars.grad)
    print(iter_ind, loss)
    print(vars.data)
    print(vars.grad)
    vars.grad.zero_()

print('============')
print(sigmoid(vars))
