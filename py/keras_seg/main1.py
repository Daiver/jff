import cv2
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import keras
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import Input, Activation, Concatenate
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Lambda, Add
from keras import optimizers

from focal_loss import focal_loss

inputShape = np.array([32, 32, 3])

def mkModel():
    model = Sequential()
    model.add(Conv2D(15, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(15, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(15, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(1, kernel_size=(1, 1), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model

def mkData(nSamples):
    data = []
    labels = []
    for ind in xrange(nSamples):
        img = np.zeros(inputShape, dtype=np.float32)
        #img = np.random.uniform(0, 1, size=inputShape)
        label = np.zeros((inputShape[0], inputShape[1], 1), dtype=np.float32)

        nRects = random.randint(0, 2)
        for _ in xrange(nRects):
            color = (random.uniform(0.0, 1.0), random.uniform(0, 1.0), random.uniform(0, 1.0))
            pt1 = (random.randrange(0, inputShape[0]), random.randrange(0, inputShape[1]))
            pt2 = (random.randrange(0, inputShape[0]), random.randrange(0, inputShape[1]))
            cv2.rectangle(img, pt1, pt2, color, random.randint(1, 1))

        nCircles = random.randint(0, 2)
        for _ in xrange(nCircles):
            circleSize = random.randint(2, 3)
            color = (random.uniform(0.0, 1.0), random.uniform(0, 1.0), random.uniform(0, 1.0))
            circleCenter = (
                    random.randrange(2*circleSize, inputShape[0] - 2*circleSize),
                    random.randrange(2*circleSize, inputShape[1] - 2*circleSize))
            cv2.circle(img  , circleCenter, circleSize, color, circleSize)
            cv2.circle(label, circleCenter, circleSize, 1.0, circleSize)

        data.append(img)
        labels.append(label)

    return np.array(data), np.array(labels)

data, labels = mkData(500)
testData, testLabels = mkData(20)
print data.shape, labels.shape
'''for img, l, in zip(data, labels):
    cv2.imshow('img', img)
    cv2.imshow('l', l)
    cv2.waitKey()'''

model = mkModel()
optimizer = optimizers.SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
#optimizer = 'adam'
model.compile(
        loss='binary_crossentropy', 
        #loss=[focal_loss(alpha=2, gamma=2)],
        metrics=['accuracy'],
        optimizer=optimizer)
model.fit(
        data, labels,
        validation_data = (testData, testLabels),
        epochs=50
        )

predicted = model.predict(testData)
predicted[predicted >= 0.5] = 1.0
predicted[predicted <  0.5] = 0.0
#predicted = model.predict_classes(testData)
for img, l, p in zip(testData, testLabels, predicted):
    img = cv2.pyrUp(cv2.pyrUp(img))
    l = cv2.pyrUp(cv2.pyrUp(l))
    p = cv2.pyrUp(cv2.pyrUp(p))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'IMG', (0, 32*4 - 4), font, 1, (0.0, 0.8, 0.2), 2)
    cv2.putText(l, 'REAL', (0, 32*4 - 4), font, 1, 1.0, 2)
    cv2.putText(p, 'PREDICTED', (0, 32*4 - 4), font, 1, 1.0, 2)

    cv2.imshow('img', img)
    cv2.imshow('l', l)
    cv2.imshow('p', p)
    cv2.waitKey()

#for x in dir(model):
#    print x
#print model.layers[0].get_weights()

#cv2.imshow('', model.predict(
#        img.reshape(1, inputShape[0], inputShape[1], 1)
#    ).reshape(inputShape[0], inputShape[1]))
#cv2.waitKey()

#cv2.imshow('', img)
#cv2.waitKey()


