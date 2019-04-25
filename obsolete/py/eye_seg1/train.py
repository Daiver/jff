import random
random.seed(42)
import numpy as np
np.random.seed(42)
import cv2
import os
import json

import keras
from keras.preprocessing import image
from keras.applications.resnet50 import decode_predictions
from keras.models import Sequential, Model

from keras import optimizers
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

from lrdecays import *
from focal_loss import *
from losses import *

from loaddata import *
from hourglass import mkHourglass
from models import *
from augmentation import augmentSegmentation, defAugSettings

DATA_ROOT = 'data/'
#targetShape = (64, 64)
#targetShape = (96, 96)
targetShape = (128, 128)
targetShape = (192, 192)


if __name__ == '__main__':

    preprocess_input = keras.applications.resnet50.preprocess_input

    data, labels, _ = loadData(targetShape, preprocess_input)
    nFinalOutputs = labels.shape[-1]

    valThreshold = 480
    trainData   = data  [:-valThreshold]
    trainLabels = labels[:-valThreshold]
    testData    = data  [-valThreshold:]
    testLabels  = labels[-valThreshold:]

    nAugTimes = 3
    augSet = defAugSettings()
    trainData, trainLabels = augmentSegmentation(
            augSet, [trainData, trainLabels], True, nAugTimes)

    print data.shape, labels.shape
    print trainData.shape, trainLabels.shape
    print testData.shape, testLabels.shape
    print 'nFinalOutputs', nFinalOutputs

    #model = mkModel3(targetShape, nFinalOutputs)
    #model = mkHourglass(targetShape, 16, nFinalOutputs=nFinalOutputs)
    #model = mkHourglass(targetShape, 32, nFinalOutputs=nFinalOutputs)
    model = mkHourglass(targetShape, 64, nFinalOutputs=nFinalOutputs)
    #model = mkHourglass(targetShape, 128, nFinalOutputs=nFinalOutputs)

    #learningRate = 0.05
    learningRate = 0.1
    #learningRate = 0.0003
    #learningRate = 0.2
    #learningRate = 1

    optimizer = optimizers.SGD(
            lr=learningRate, decay=0e-6, momentum=0.9, nesterov=True)
    #optimizer = optimizers.Adam(lr=learningRate)
    #optimizer = 'adam'
    model.compile(
        #loss='mse', 
        loss=weightedBinaryCrossentropy([1.0 , 1.0]), 
        #loss='binary_crossentropy', 
        #loss=[focal_loss(alpha=2, gamma=2)],
        #metrics=['accuracy'],
        optimizer=optimizer
        )

    #epochs = 10
    #epochs = 20
    #epochs = 30
    #epochs = 50
    epochs = 100
    #epochs = 1000

    decayFunc = expDecay(initial_lrate=learningRate, epochs_drop=5, drop=0.8)
    #decayFunc = linearDecay(initial_lrate=learningRate, nIters=epochs)
    #decayFunc = sqrtDecay(initial_lrate=initialLearningRate, nIters=epochs)
    lrate = LearningRateScheduler(decayFunc)
    callbacks = [
            keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True),
            keras.callbacks.ModelCheckpoint('dump/checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1),
            #lrate,
            ReduceLROnPlateau(),
            ]

    #sampleWeight = np.copy(trainLabels)
    #sampleWeight[sampleWeight <  0.5] = 0.009959
    #sampleWeight[sampleWeight >= 0.5] = 0.99


    nModelOuts = len(model.outputs)
    model.fit(
        trainData, [trainLabels] * nModelOuts,
        validation_data = (testData, [testLabels] * nModelOuts),
        epochs=epochs,
        batch_size=16,
        #batch_size=32,
        callbacks=callbacks
        )
    model.save('dump/model.h5')

