import numpy as np
from torch.autograd import Variable
import torch.onnx
import torchvision

dummy_input = Variable(torch.randn(1, 3))

linearLayer = torch.nn.Linear(3, 2)
linearLayer.weight = torch.nn.Parameter(torch.Tensor([
    [1, 2, 3],
    [4, 5, 6]
]))

linearLayer.bias = torch.nn.Parameter(torch.Tensor([
    10, 12
]))

print(linearLayer.weight)
print(linearLayer.bias)
model = torch.nn.Sequential(linearLayer)

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(2) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "torch_linear.proto", verbose=True, input_names=input_names, output_names=output_names)
