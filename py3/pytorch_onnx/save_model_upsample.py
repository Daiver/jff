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
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        return x

model = ConvNet()

model = model.to('cpu')

dummy_input = Tensor(1, 1, 28, 28)

torch.onnx.export(
    model, dummy_input, 
    "torch_conv_simple_upsample.proto", verbose=True)

inp = Tensor([1, 2, 3, 4]).reshape(1, 1, 2, 2)
print(inp)
print(F.upsample(inp, scale_factor=2))

