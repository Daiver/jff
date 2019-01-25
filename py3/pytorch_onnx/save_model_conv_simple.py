import numpy as np
from torch.autograd import Variable
import torch.onnx
import torchvision

#dummy_input = Variable(torch.randn(1, 1, 3, 3))
dummy_input = Variable(torch.from_numpy(np.array([
    range(1, 73)
], dtype=np.float32).reshape(2, 1, 6, 6)))

convLayer = torch.nn.Conv2d(1, 1, kernel_size=[3, 3], stride=[3, 3], padding=2, bias=True)
convLayer.weight = torch.nn.Parameter(torch.Tensor([
    range(2, 52)
]).reshape(2, 1, 5, 5))
#convLayer.weight = torch.nn.Parameter(torch.Tensor([
#    [1, 3, 5],
#    [7, 9, 11],
#    [13, 15, 17]
#]).reshape(1, 1, 3, 3))

convLayer.bias = torch.nn.Parameter(torch.Tensor([
    0, 0
]))

print(convLayer.weight)
print(convLayer.bias)
model = torch.nn.Sequential(convLayer)

print(model(dummy_input))

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(2) ]
output_names = [ "output1" ]

#torch.onnx.export(
    #model, dummy_input, 
    #"torch_conv_simple.proto", verbose=True, input_names=input_names, output_names=output_names)
