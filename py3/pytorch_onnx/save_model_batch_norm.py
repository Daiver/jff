import numpy as np
import torch.nn.functional as F

from torch import Tensor
from torch.autograd import Variable
import torch.onnx
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(2, eps=0.1, momentum=1)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        return x

torch.set_printoptions(precision=10)


model = ConvNet()
model.eval()

dummy_input = Variable(Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3))

conv = model.conv1
conv.weight = nn.Parameter(Tensor([
    0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 2, 0, 0, 0, 0]).view(2, 1, 3, 3))
conv.bias = nn.Parameter(Tensor([0, -1]))

bn = model.bn1
bn.weight = nn.Parameter(Tensor([1.1, 0.2]))
bn.bias = nn.Parameter(Tensor([0.5, 0]))
bn.running_mean = Tensor([2, 1])
bn.running_var = Tensor([0.5, 2])

model.eval()
output = model(dummy_input)
print('output')
print(output)

torch.onnx.export(
    model, dummy_input, 
    "torch_conv_batchnormalization.proto", verbose=True)
