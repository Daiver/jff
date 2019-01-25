import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import cv2
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

import train

if __name__ == '__main__':
    modelName = "dump/model.h5"
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
    data, labels, images = train.loadData(train.targetShape, keras.applications.resnet50.preprocess_input)
    print data.shape, labels.shape

    lowerBound = 480
    index = 1
    indexOfInterest = lowerBound - index
    data   = np.array([data  [-indexOfInterest]])
    images = np.array([images[-indexOfInterest]])

    for lIndex, layer in enumerate(model.layers):
        if layer.name.find('act') < 0 :
            continue
        print lIndex, layer.name
        inp = model.input
        out = layer.output
        m   = Model(inputs=[inp], outputs=[out])
        predicted = m.predict(data)
        print predicted.shape
        #predicted = predicted.reshape((-1, train.targetShape[0], train.targetShape[1]))

        i = 0
        img = images[i]
        img = cv2.pyrUp(img)
        cv2.imshow('img', img)
        pMin = np.min(predicted)
        pMax = np.max(predicted)
        predicted = (predicted - pMin)/(pMax - pMin)
        for c in xrange(predicted.shape[3]):
            tmp = predicted[i, :, :, c]
            tmp = cv2.resize(tmp, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('channel%s' % c, tmp)
            if c > 32:
                break
        cv2.waitKey()
        #cv2.destroyAllWindows()
