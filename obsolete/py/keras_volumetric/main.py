import numpy as np
np.random.seed(42)
import random

import keras
from keras.layers import Activation, Conv3D, MaxPooling3D
from keras.layers import Dense, Flatten, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model, Sequential

inputShape = [5, 5, 5, 1]

def mkData(n1Classes, n2Classes):
    data   = np.zeros([n1Classes + n2Classes, ] + inputShape, dtype=np.float32)
    labels = np.zeros([n1Classes + n2Classes, ], dtype=np.float32)
    for i in xrange(n1Classes):
        x = random.randint(0, inputShape[0] - 1)
        y = random.randint(0, inputShape[1] - 1)
        z = random.randint(0, inputShape[2] - 1)
        sample = np.zeros(inputShape)
        sample[x, y, z] = 1
        if x >= inputShape[0] - 1:
            x -= 1
        else:
            x += 1
        sample[x, y, z] = 1
        data[i] = sample
        labels[i] = 0

    for i in xrange(n2Classes):
        x = random.randint(0, inputShape[0] - 1)
        y = random.randint(0, inputShape[1] - 1)
        z = random.randint(0, inputShape[2] - 1)
        sample = np.zeros(inputShape)
        sample[x, y, z] = 1
        if y >= inputShape[1] - 1:
            y -= 1
        else:
            y += 1
        sample[x, y, z] = 1
        data[i + n1Classes] = sample
        labels[i + n1Classes] = 1

    return data.reshape([-1, ] + inputShape), labels 

if __name__ == '__main__':
    data, labels = mkData(100, 100)
    testData, testLabels = mkData(20, 20)

    model = Sequential()
    model.add(Conv3D(8, kernel_size=(3, 3, 3), padding='same', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(strides=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(data, labels, 
            epochs=100, 
            batch_size=1, 
            validation_data=(testData, testLabels),
            verbose=True)
