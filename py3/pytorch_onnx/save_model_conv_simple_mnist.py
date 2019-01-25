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
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.fc = nn.Linear(28*28*16, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = out.view((out.size(0), -1))
        out = self.fc(out)
        return out

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

batch_size = 64
test_batch_size = 1000
lr = 1e-3
momentum = 0.5
#epochs = 1
#epochs = 20
epochs = 200

torch.manual_seed(42)
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, pin_memory=True, num_workers=1)

model = ConvNet().to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for epoch in range(1, epochs + 1):
    print(epoch, '/', epochs)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

model = model.to('cpu')
#dummy_input = Variable(Tensor(1, 1, 28, 28))
batch = next(iter(test_loader))[0]
print(batch.shape)
dummy_input = Variable(batch[0]).reshape(1, 1, 28, 28)
print(dummy_input)
print(model(dummy_input))


input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(2) ]
output_names = [ "output1" ]


torch.onnx.export(
    model, dummy_input, 
    "torch_conv_simple_mnist1_bad.proto", verbose=True)
