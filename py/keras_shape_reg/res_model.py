import numpy as np
np.random.seed(42)
#from input_shape import input_shape

import keras
from keras.datasets import mnist
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Lambda, add
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers

from keras import backend as K

def shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        #kernel_regularizer = l2(1e-5)
        #kernel_regularizer = l2(1e-6)
        kernel_regularizer = None
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(2, 2),
                          #kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=kernel_regularizer)(input)

    return add([shortcut, residual])

def residualBlock(nFilters, init_strides=(1, 1)):
    use_bias = True
    kernel_initializer = "he_normal"
    #kernel_regularizer = l2(1e-4)
    #kernel_regularizer = l2(1e-5)
    #kernel_regularizer = l2(1e-6)
    kernel_regularizer = None#l2(1e-4)

    def f(x):
        bn1   = BatchNormalization()(x)
        act1  = Activation('relu')(bn1)
        #act1  = LeakyReLU(alpha=0.001)(bn1)
        conv1 = Conv2D(nFilters, kernel_size=(3, 3), 
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                strides=init_strides,
                padding='same', use_bias=False)(act1)

        bn2   = BatchNormalization()(conv1)
        act2  = Activation('relu')(bn2)
        #act2  = LeakyReLU(alpha=0.001)(bn2)
        conv2 = Conv2D(nFilters, kernel_size=(3, 3),
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                padding='same', use_bias=use_bias)(act2)

        conv2 = Dropout(0.3)(conv2)

        return shortcut(x, conv2)
    return f

def mkResnet(nFeats, input_shape, includeTop=True):
    #nFilters = 64
    nFilters = 32
    #nFilters = 16

    #convReg = l2(1e-4)
    convReg = None

    inp = Input(input_shape)
    x = inp
    x = Conv2D(int(1.5 ** 0 * nFilters), kernel_size=(7, 7), strides=(2, 2), 
            padding='same', kernel_regularizer=convReg)(x)
    x = Activation('relu')(x)
    #x = LeakyReLU(alpha=0.001)(x)
    #x = BatchNormalization()(x)
    #x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = residualBlock(int(1.5 ** 0 * nFilters))(x)
    x = residualBlock(int(1.5 ** 1 * nFilters), (2, 2))(x)
    #x = residualBlock(int(1.5 ** 1 * nFilters))(x)
    x = residualBlock(int(1.5 ** 2 * nFilters), (2, 2))(x)
    #x = residualBlock(int(1.5 ** 2 * nFilters))(x)
    x = residualBlock(int(1.5 ** 3 * nFilters), (2, 2))(x)
    #x = residualBlock(int(1.5 ** 3 * nFilters))(x)
    #x = residualBlock(int(1.5 ** 3 * nFilters))(x)
    x = residualBlock(int(1.5 ** 4 * nFilters), (2, 2))(x)
    #x = residualBlock(int(1.5 ** 4 * nFilters))(x)
    x = residualBlock(int(1.5 ** 5 * nFilters), (2, 2))(x)

    if not includeTop:
        model = Model(inputs=inp, outputs=x)
        return model

    x = Flatten()(x)
    #x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    r1 = Dense(nFeats, activation='linear')(x)
    model = Model(inputs=inp, outputs=r1)
    return model

