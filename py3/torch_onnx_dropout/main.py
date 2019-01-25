import torch
import random
random.seed(42)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.dropout = torch.nn.Dropout2d(0.5)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = x * 2
        x = self.dropout(x)
        x = x + 1
        return x

print('torch version', torch.__version__)

model = Model()
model.train(True)
dummy_input = torch.ones(1, 4, 1, 1)
# dummy_input = torch.ones(1, 4)
print(model(dummy_input))
print(model(dummy_input))
print(model(dummy_input))
print(model(dummy_input))
print(model(dummy_input))
model.train(False)
print(model(dummy_input))
torch.onnx.export(model, dummy_input, 'tmp.proto', verbose=True)

