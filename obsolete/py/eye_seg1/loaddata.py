import numpy as np
import cv2
import json

def segmentationMaskFromColors(classColorCodes, label):
    shape = label.shape
    assert(shape[2] == 3)
    nSubClasses = len(classColorCodes)
    res = np.zeros((shape[0], shape[1], nSubClasses), dtype=np.float32)
    for i, c in enumerate(classColorCodes):
        #res[label] = 1.0
        initialMask = label == c
        if np.any(initialMask) == False:
            continue
        mask = np.all(initialMask, axis=2)
        res[mask, i] = 1.0
    return res
    
def loadData(
        targetShape, 
        preprocess_input,
        dataFilePath='data/data.txt'
        ):

    with open(dataFilePath) as f:
        dataDesc = json.load(f)
    samples = dataDesc['samples']
    nSubClasses = len(samples[0]['paths2Label'])
    print 'nSamples', len(samples)

    allowAllSubdirs = True
    allowedSubdirs = ['TrainingPose']

    imgsNames = [x['path2Img'] for x in samples   if allowAllSubdirs or x['subdir'] in allowedSubdirs]
    lblsNames = [x['paths2Label'] for x in samples if allowAllSubdirs or x['subdir'] in allowedSubdirs]

    nItems = len(imgsNames)
    assert(nItems == len(lblsNames))

    images = []

    data   = np.zeros((nItems, targetShape[0], targetShape[1], 3), dtype=np.float32)
    labels = np.zeros((nItems, targetShape[0], targetShape[1], nSubClasses), dtype=np.float32)
    for i, (imgName, lblName) in enumerate(zip(imgsNames, lblsNames)):
        img   = cv2.imread(imgName)
        img   = cv2.resize(img, targetShape)
        images.append(img)
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	label = np.zeros((targetShape[0], targetShape[1], nSubClasses), dtype=np.float32)
        for chnlInd, lblChannelName in enumerate(lblName):
            l = cv2.imread(lblChannelName, 0)
            l = cv2.resize(l, targetShape)
            l = l.astype(np.float32) / 255.0
            label[:, :, chnlInd] = l

        data[i, :, :, :] = img
        labels[i] = label.reshape((targetShape[0], targetShape[1], nSubClasses))

    data = preprocess_input(data)

    nItems    = data.shape[0]*data.shape[1]*data.shape[2]*data.shape[3] 
    lbls2Sum  = labels[labels > 0.1].astype(np.float64) / nItems
    nPositive = lbls2Sum.sum()
    nNegative = 1.0 - nPositive
    print 'nItems', nItems, 'nPos', nPositive, 'nNeg', nNegative
    return data, labels, images


