import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
import cv2
import torchvision
import json
import os

from models import SegNet1
import unets
import unets2
import losses

class Helen2Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Helen2Dataset, self).__init__()
        path2Helen = '/home/daiver/R3DS/Kirill/FaceSegmentation1/eye_seg1/'
        path2HelenDataTxt = os.path.join(path2Helen, 'data/data.txt')
        dataJson = json.load(open(path2HelenDataTxt))
        samples = dataJson['samples']
        self.imgs = []
        self.labels = []
        for sample in samples:
            path2Img = os.path.join(path2Helen, sample['path2Img'])
            path2Lbl = os.path.join(path2Helen, sample['paths2Label'][0])
            img = cv2.imread(path2Img)
            label = cv2.imread(path2Lbl, 0)
            label = (label.astype(np.float32) / 255.0)
            self.imgs.append(torchvision.transforms.ToTensor()(img).float())
            self.labels.append(torch.from_numpy(label).long())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ind):
        return (self.imgs[ind], self.labels[ind])

def train(model, device, trainLoader, oprimizer, epoch):
    model.train()
    for batchInd, (data, target) in enumerate(trainLoader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #output = F.log_softmax(output, dim=1)
        output = output.view(-1, output.shape[2], output.shape[3])
        #loss = F.nll_loss(output, target)
        criterion = losses.LossBinary(0.1)
        #loss = F.binary_cross_entropy_with_logits(output, target.float())
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()

        if batchInd % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batchInd * len(data), len(trainLoader.dataset),
                100. * batchInd / len(trainLoader), loss.item()))


#batchSize = 32
#batchSize = 16
batchSize = 8
#batchSize = 4
#batchSize = 2

device = 'cuda'
#device = 'cpu'
lr = 1e-4
#lr = 3e-5
momentum = 0.9
epochs = 200
#epochs = 20
#epochs = 1

#model = SegNet1()
#model = unets.UNet11(pretrained=True)
#model = unets.UNet16(pretrained=True)
#model = unets.AlbuNet(pretrained=True)
model = unets2.LinkNet34(pretrained=True)

#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimizer = torch.optim.Adam(model.parameters())


dataset = Helen2Dataset()
print(len(dataset))

trainLoader = torch.utils.data.DataLoader(
        dataset, batch_size=batchSize, shuffle=True, pin_memory=True, num_workers=10)

model = model.to(device)

for epoch in range(1, epochs + 1):
    print(epoch, '/', epochs)
    try:
        train(model, device, trainLoader, optimizer, epoch)
        #test(model, device, test_loader)
        torch.save(model.state_dict(), 'model.pt')
    except KeyboardInterrupt:
        print('Ctrl+C, saving snapshot')
        torch.save(model.state_dict(), 'model.pt')
        exit(0)



