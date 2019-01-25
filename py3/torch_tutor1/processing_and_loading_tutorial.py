import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

landmarksFrame = pd.read_csv('faces/face_landmarks.csv')

n = 65
imgName = landmarksFrame.iloc[n, 0]
landmarks = landmarksFrame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype(np.float32).reshape(-1, 2)

print('Image name: {}'.format(imgName))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 landmarks: {}'.format(landmarks[:4]))

def showLandmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)
    #plt.pause(10.001)

#plt.figure()
#showLandmarks(io.imread(os.path.join('faces/', imgName)), landmarks)
#plt.show()


class FaceLandmarksDataset(Dataset):
    def __init__(self, csvFileName, rootDir, transform=None):
        self.landmarksFrame = pd.read_csv(csvFileName)
        self.rootDir = rootDir
        self.transform = transform

    def __len__(self):
        return len(self.landmarksFrame)

    def __getitem__(self, idx):
        imgName = os.path.join('faces', self.landmarksFrame.iloc[idx, 0])
        image = io.imread(imgName)
        landmarks = self.landmarksFrame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype(np.float32).reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


faceDataset = FaceLandmarksDataset('faces/face_landmarks.csv', 'faces')

fig = plt.figure()

for i in range(len(faceDataset)):
    sample = faceDataset[i]
    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    showLandmarks(**sample)

    if i == 3:
        plt.show()
        break


class Rescale():
    def __init__(self, outSize):
        self.outSize = outSize

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.outSize, int):
            if h > w:
                newH, newW = self.outSize * h / w, self.outSize
            else:
                newH, newW = self.outSize, self.outSize * w / h
        else:
            newH, newW = self.outSize

        newH, newW = int(newH), int(newW)

        img = transform.resize(image, (newH, newW))
        landmarks = landmarks * [newW / w, newH / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop():
    def __init__(self, outSize):
        if isinstance(outSize, int):
            outSize = (outSize, outSize)
        self.outSize = outSize

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        newH, newW = self.outSize

        top  = np.random.randint(0, h - newH)
        left = np.random.randint(0, w - newW)
        image = image[top: top + newH, left: left + newW, :]
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor():
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}


scaler  = Rescale(256)
cropper = RandomCrop(128)

composed = transforms.Compose([Rescale(256), RandomCrop(224)])

fig = plt.figure()
sample = faceDataset[65]

for i, trnsfrm in enumerate([scaler, cropper, composed]):
    transformedSample = trnsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(trnsfrm).__name__)
    showLandmarks(**transformedSample)

composed = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])
transformedDataset = FaceLandmarksDataset('faces/face_landmarks.csv', 'faces', transform=composed)

dataloader = DataLoader(transformedDataset, batch_size=4, shuffle=True, num_workers=4)

def showLandmarksBatch(sampleBatched):
    imagesBatch, landmarksBatch = sampleBatched['image'], sampleBatched['landmarks']
    batchSize = imagesBatch.shape[0]
    imSize = imagesBatch.size(2)
    grid = utils.make_grid(imagesBatch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
for i, sampleBatched in enumerate(dataloader):
    print(i, sampleBatched['image'].size(), sampleBatched['landmarks'].size())
    if i == 3:
        plt.figure()
        showLandmarksBatch(sampleBatched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break








