import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import cv2
import time

import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import decode_predictions
from keras.models import Sequential, Model, load_model
from keras.regularizers import l2
from keras.layers import Input, Activation, Concatenate
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Lambda, Add
from keras import optimizers
from keras.utils.generic_utils import CustomObjectScope

from losses import *
from focal_loss import *

from loaddata import loadData
import train

if __name__ == '__main__':
    modelName = "dump/model.h5"
    modelName = "dump/checkpoint.h5"
    print 'reading...', modelName
    #modelName = "dump/model_bak.h5"
    with CustomObjectScope({
        'relu6': keras.applications.mobilenet.relu6,
        'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D,
        'f' : weightedBinaryCrossentropy([1, 1])
        }):
        model = load_model(modelName)
        inp = model.inputs[0]
        out = model.outputs[-1]
        model = Model(inputs=[inp], outputs=[out])
    print model.summary()
    dataDir = "data/data.txt"
    #dataDir = "data.txt"
    data, labels, images = loadData(train.targetShape, keras.applications.resnet50.preprocess_input, dataDir)
    print data.shape, labels.shape

    images = np.array(images, dtype=np.uint8)

    lowerBound = 480#480 validation images in real
    #lowerBound = 8# validation images in real
    data   = data  [-lowerBound:]
    labels = labels[-lowerBound:]
    images = images[-lowerBound:]
    print data.shape

    nSubClasses = labels.shape[-1]

    #tmp = data.shape[1]
    #data  [:, 0:tmp-10] = data  [:, 10:tmp]
    #labels[:, 0:tmp-10] = labels[:, 10:tmp]
    #images[:, 0:tmp-10] = images[:, 10:tmp]

    startTime = time.time()
    predicted = model.predict(data)
    print 'Prediction elapsed', time.time() - startTime
    predicted = predicted.reshape((-1, train.targetShape[0], train.targetShape[1], nSubClasses))
    labels    = labels.reshape((-1, train.targetShape[0], train.targetShape[1], nSubClasses))
    for i in range(data.shape[0]):
        img = images[i]
        label = labels[i]
        predict = predicted[i]

        def mkLabel3Channel(label):
            res = np.zeros((label.shape[0], label.shape[1], 3), dtype=label.dtype)
            res[:, :, 0] = label[:, :, 0]
            res[:, :, 1] = label[:, :, 1]
            return res

        #labelChannelInd = 0
        #label   = label  [:, :, labelChannelInd]
        #predict = predict[:, :, labelChannelInd]
        label = mkLabel3Channel(label)
        predict = mkLabel3Channel(predict)

        nLabelChannels = 1 if len(label.shape) == 2 or label.shape[-1] == 1 else 3

        #predict[predict >  0.5] = 1
        #predict[predict <= 0.5] = 0

        img = cv2.pyrUp(img)
        label = cv2.pyrUp(label)
        predict = cv2.pyrUp(predict)

        overlayR = np.copy(img).astype(np.float32) + label.astype(np.float32).reshape((img.shape[0], img.shape[1], nLabelChannels)) * 255
        overlayR[overlayR > 255] = 255
        overlayR = overlayR.astype(np.uint8)

        overlayP = np.copy(img).astype(np.float32) + predict.astype(np.float32).reshape((img.shape[0], img.shape[1], nLabelChannels)) * 255
        overlayP[overlayP > 255] = 255
        overlayP = overlayP.astype(np.uint8)

        intersect = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
        #intersect[:,:, 2] = predict.reshape((label.shape[0], label.shape[1]))
        #intersect[:,:, 1] = label.reshape((label.shape[0], label.shape[1]))

        textOffset = img.shape[1] * 2 - 4

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img     , 'IMG'      , (0, textOffset), font, 1, (0.0, 0.8, 0.2), 2)
        cv2.putText(label   , 'REAL'     , (0, textOffset), font, 1, 1.0, 2)
        cv2.putText(predict , 'PREDICTED', (0, textOffset), font, 1, 1.0, 2)
        cv2.putText(overlayR, 'R_OVERLAY', (0, textOffset), font, 0.8, 1.0, 2)
        cv2.putText(overlayP, 'P_OVERLAY', (0, textOffset), font, 0.8, 1.0, 2)

        imgSide = img.shape[0]
        img2Show = np.zeros((imgSide * 2, imgSide * 3, 3), dtype=np.uint8)
        img2Show[0:imgSide, 0:imgSide] = img
        img2Show[0:imgSide:, imgSide:2*imgSide] = overlayR
        img2Show[0:imgSide:, 2*imgSide:3*imgSide] = overlayP
        img2Show[imgSide:2*imgSide:, 1*imgSide:2*imgSide] = label.reshape((imgSide, imgSide  , nLabelChannels)) * 255
        img2Show[imgSide:2*imgSide:, 2*imgSide:3*imgSide] = predict.reshape((imgSide, imgSide, nLabelChannels)) * 255
        img2Show[imgSide:2*imgSide, 0:imgSide] = intersect * 255
        cv2.imwrite('to_show/%s.png' % i, img2Show)
        cv2.imshow('', img2Show)

        #cv2.imshow('img', img)
        #cv2.imshow('overlayR', overlayR)
        #cv2.imshow('overlayP', overlayP)
        #cv2.imshow('lbl', label)
        #cv2.imshow('pred', predict)
        cv2.waitKey(100)
