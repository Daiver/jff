import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
random.seed(42)
import numpy as np
np.random.seed(42)
import cv2

import keras
from keras.preprocessing import image
from keras.applications.resnet50 import decode_predictions
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import *

from losses import *

imgSize = (32, 32)

img = np.zeros(imgSize, dtype=np.float32)
cv2.circle(img, (imgSize[0]//2, imgSize[1]//2), 5, 1.0, 1)


model = Sequential()
model.add(Conv2D(10, kernel_size=(3, 3), input_shape=(imgSize[0], imgSize[1], 1), padding='same'))
model.add(Conv2D(10, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(1, kernel_size=(3, 3), padding='same'))
model.add(Activation('sigmoid'))

model.compile(
        optimizer='adam', 
        #loss='binary_crossentropy')
        loss=weightedBinaryCrossentropy([0.5, 10.5]))

data   = img.reshape((1, imgSize[0], imgSize[1], 1))
labels = img.reshape((1, imgSize[0], imgSize[1], 1))
model.fit(data, labels, epochs=10)

pred = model.predict(data)
pred = pred.reshape((imgSize))
cv2.imshow('', img)
cv2.imshow('1', pred)
cv2.waitKey()
