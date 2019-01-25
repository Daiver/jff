import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()

def imshow(inp, title=None, pauseTimeOut=0.001):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(pauseTimeOut)

IMG_SIDE = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(IMG_SIDE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIDE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}

data_dir = 'hymenoptera_data'
image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
}
dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=4,
            shuffle=True, num_workers=4)
        for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()
print('USE GPU:', use_gpu)

inputs, classes = next(iter(dataloaders['train']))
examplesGrid = torchvision.utils.make_grid(inputs)
imshow(examplesGrid, [class_names[x] for x in classes])


def trainModel(model, criterion, optimizer, scheduler, numEpochs=25):
    since = time.time()
    bestModelWeights = copy.deepcopy(model.state_dict())
    bestAcc = 0.0

    for epoch in range(numEpochs):
        print('-'*10)
        print("Epoch {}/{}".format(epoch + 1, numEpochs))
        print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
        
            runningLoss = 0.0
            runningCorrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                inputs = Variable(inputs)
                labels = Variable(labels)
                
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                runningLoss += loss.data[0] * inputs.size(0)
                runningCorrects += torch.sum(preds == labels.data)

            epochLoss = runningLoss / dataset_sizes[phase]
            epochAcc  = runningCorrects / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epochLoss, epochAcc))

            if phase == 'val' and epochAcc > bestAcc:
                bestAcc = epochAcc
                bestModelWeights = copy.deepcopy(model.state_dict())
        print()

    timeElapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(timeElapsed // 60, timeElapsed % 60))
    print('Best val Acc {:4f}'.format(bestAcc))
    model.load_state_dict(bestModelWeights)
    return model


def visualizeModel(model, numImages=6):
    wasTraining = model.training
    model.eval()
    imagesSoFar = 0
    fig = plt.figure()

    for i, (inputs, labels) in enumerate(dataloaders['val']):
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size(0)):
            imagesSoFar += 1
            nCols = 2
            ax = plt.subplot(numImages // nCols, nCols, imagesSoFar)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])
            
            if imagesSoFar == numImages:
                model.train(mode=wasTraining)
                return
    model.train(mode=wasTraining)

modelFt = models.resnet18(pretrained=True)
#modelFt = models.resnet18(pretrained=False)
for param in modelFt.parameters():
    param.requires_grad = False

numFeatures = modelFt.fc.in_features
modelFt.fc = nn.Linear(numFeatures, 2)


if use_gpu:
    modelFt = modelFt.cuda()

criterion = nn.CrossEntropyLoss()

#optimizerFt = optim.SGD(modelFt.parameters(), lr=0.001, momentum=0.9)
#expLrScheduler = lr_scheduler.StepLR(optimizerFt, step_size=7, gamma=0.1)
#numEpochs=25

#optimizerFt = optim.SGD(modelFt.parameters(), lr=0.001, momentum=0.9)
optimizerFt = optim.SGD(modelFt.fc.parameters(), lr=0.001, momentum=0.9)
expLrScheduler = lr_scheduler.StepLR(optimizerFt, step_size=7, gamma=0.1)
numEpochs=25

modelFt = trainModel(modelFt, criterion, optimizerFt, expLrScheduler, numEpochs=numEpochs)

visualizeModel(modelFt, 12)
plt.pause(20)
