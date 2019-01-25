import numpy as np
np.random.seed(42)
import random
random.seed(42)

import matplotlib.pyplot as plt

import keras
from keras import optimizers
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import Input, Activation, Concatenate
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Lambda, Add

def genDataAnsLabels(nSamples):
    radius  = 1
    borders = [-1.3, 1.3]
    data = np.zeros((nSamples, 2))
    for i in xrange(nSamples):
        x1 = random.uniform(borders[0], borders[1])
        x2 = random.uniform(borders[0], borders[1])
        data[i, :] = [x1, x2]

    labels = np.zeros((nSamples))

    for i, (x1, x2) in enumerate(data):
        labels[i] = 1 if x1**2 + x2**2 <= radius**2 else 0

    return data, labels

def drawDataWithLabels(data, labels, data2=None, labels2=None):
    labels = labels.reshape(-1)
    pos = data[labels >= 0.5, :]
    neg = data[labels <  0.5, :]
    print 'n pos', len(pos), 'n neg', len(neg)
    plt.plot(pos[:, 0], pos[:, 1], 'go')
    plt.plot(neg[:, 0], neg[:, 1], 'ro')
    if data2 is not None:
        labels2 = labels2.reshape(-1)
        pos = data2[labels2 >= 0.5, :]
        neg = data2[labels2 <  0.5, :]
        plt.plot(pos[:, 0], pos[:, 1], 'gx', markersize=10)
        plt.plot(neg[:, 0], neg[:, 1], 'rx', markersize=10)

    plt.show(block=True)

def denseBlock(x):
    print K.int_shape(x)
    d = Dense(1)(x)
    a = Activation('relu')(d)
    return Concatenate()([x, a])


def residual(x):
    d = Dense(2)(x)
    a = Activation('relu')(d)
    return Add()([x, a])
    #b = BatchNormalization()(a)
    #return Add()([x, b])

if __name__ == '__main__':
    nSamplesTrain = 5000
    dataTrain, labelsTrain = genDataAnsLabels(nSamplesTrain)
    nSamplesTest = 2000
    dataTest, labelsTest = genDataAnsLabels(nSamplesTest)

    inp = Input((2,))
    x = inp
    x = denseBlock(x)
    x = denseBlock(x)
    x = denseBlock(x)
    x = denseBlock(x)
    x = denseBlock(x)
    x = denseBlock(x)
    x = denseBlock(x)
    x = denseBlock(x)
    x = denseBlock(x)
    x = denseBlock(x)
    #x = residual(x)
    #x = residual(x)
    #x = Dense(2)(x)
    #x = Activation('relu')(x)
    #x = Dense(2)(x)
    #x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=inp, outputs=x)

    model.compile(loss='mse', optimizer='adam')

    model.fit(
            dataTrain, labelsTrain, 
            batch_size=32, epochs=50, verbose=True,
            validation_data=(dataTest, labelsTest))

    #drawDataWithLabels(dataTrain, labelsTrain)
    #drawDataWithLabels(dataTest, model.predict(dataTest))
    drawDataWithLabels(dataTest, model.predict(dataTest))
    drawDataWithLabels(dataTest, labelsTest, dataTest, model.predict(dataTest))
