import numpy as np
import torch 
from torch import Tensor
from torch.autograd import Variable


class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0.0)

    @staticmethod
    def backward(ctx, outGrad):
        inp, = ctx.saved_tensors
        res = outGrad.clone()
        res[inp < 0] = 0
        return res


myRelu = MyReLU.apply

x = Variable(Tensor([1, -2, 2]), requires_grad=True)
print(x)

y1 = (10.0 * x * myRelu(x)).sum()
y1.backward()
print(y1)
print(x.grad.data)


y2 = (10.0 * x * myRelu(x)).sum()
x.grad.data.zero_()
y2.backward()
print(y2)
print(x.grad.data)
