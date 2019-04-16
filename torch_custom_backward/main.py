import torch
import torch.autograd
# import torch.nn as nn
# import torch.nn.functional as F


class MyMul(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        
        index = torch.tensor(13)
        ctx.mark_non_differentiable(index)
        return x * y, index

    @staticmethod
    def backward(ctx, grad_output, out2):
        x, y = ctx.saved_tensors
        return y * grad_output, x * grad_output, None


def main():
    x = torch.FloatTensor([2]).requires_grad_(True)
    y = torch.FloatTensor([3]).requires_grad_(True)

    # res = (x * y)
    res = MyMul.apply(x, y)[0]
    res = res * 10
    res.backward()
    print(res)
    print(x.grad)
    print(y.grad)


if __name__ == '__main__':
    main()
