import numpy as np
np.random.seed(42)
#from input_shape import input_shape

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
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers

from keras import backend as K
#from keras_contrib.layers.advanced_activations import SReLU


def mkAvgPlusMaxPool(shape):
    inp = Input(shape[1:])
    x = inp
    maxPool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    avgPool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Add()([maxPool, avgPool])
    model = Model(inputs=inp, outputs=x)
    return model

def mkDarknetLight(nFeats, input_shape, includeTop=True):
    model = Sequential()
    #nFilters = 16
    #nFilters = 32
    nFilters = 64
    #nFilters = 128

    #kernel_initializer='he_uniform'
    kernel_initializer='he_normal'
    #useMiddleBias = False
    useMiddleBias = True

    model.add(Conv2D(int(1.5**0 * nFilters), kernel_size=(5, 5), strides=(2, 2), 
        kernel_initializer=kernel_initializer,
        use_bias=True, padding='same', input_shape=input_shape))
    #model.add(LeakyReLU(alpha=.001))
    #model.add(PReLU(shared_axes=[1, 2]))
    model.add(Activation('relu'))

    model.add(mkAvgPlusMaxPool(model.output_shape))
   
    model.add(BatchNormalization())
    model.add(Conv2D(int(1.5**1 * nFilters), kernel_size=(3, 3), 
        kernel_initializer=kernel_initializer, use_bias=useMiddleBias, padding='same'))
    #model.add(LeakyReLU(alpha=.001))
    #model.add(PReLU(shared_axes=[1, 2]))
    model.add(Activation('relu'))

    model.add(mkAvgPlusMaxPool(model.output_shape))

    model.add(BatchNormalization())
    model.add(Conv2D(int(1.5**2 * nFilters), kernel_size=(3, 3), 
        kernel_initializer=kernel_initializer, use_bias=useMiddleBias, padding='same'))
    #model.add(LeakyReLU(alpha=.001))
    #model.add(PReLU(shared_axes=[1, 2]))
    model.add(Activation('relu'))

    model.add(mkAvgPlusMaxPool(model.output_shape))
    
    model.add(BatchNormalization())
    model.add(Conv2D(int(1.5**3 * nFilters), kernel_size=(3, 3), 
        kernel_initializer=kernel_initializer,use_bias=useMiddleBias, padding='same'))
    #model.add(LeakyReLU(alpha=.001))
    #model.add(PReLU(shared_axes=[1, 2]))
    model.add(Activation('relu'))

    model.add(mkAvgPlusMaxPool(model.output_shape))

    model.add(BatchNormalization())
    model.add(Conv2D(int(1.5**4 * nFilters), kernel_size=(3, 3), 
        kernel_initializer=kernel_initializer,use_bias=useMiddleBias, padding='same'))
    #model.add(LeakyReLU(alpha=.001))
    model.add(Activation('relu'))

    model.add(mkAvgPlusMaxPool(model.output_shape))

    model.add(BatchNormalization())
    model.add(Conv2D(int(1.5**5 * nFilters), kernel_size=(3, 3), 
        kernel_initializer=kernel_initializer,use_bias=useMiddleBias, padding='same'))
    model.add(Activation('relu'))

    model.add(mkAvgPlusMaxPool(model.output_shape))

    model.add(BatchNormalization())
    model.add(Conv2D(int(1.5**6 * nFilters), kernel_size=(3, 3), 
        kernel_initializer=kernel_initializer,use_bias=useMiddleBias, padding='same'))
    model.add(Activation('relu'))

    if not includeTop:
        return model

    model.add(Flatten())
    #model.add(Dropout(0.5))
    #model.add(Dense(5000, activation='relu', kernel_initializer=kernel_initializer))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(
        Dense(nFeats, activation='linear', kernel_initializer=kernel_initializer,
            #kernel_regularizer=l2(1e-5)
            ))
    #model.add(Dense(nFeats, activation='linear', kernel_regularizer=l2(0.00000001)))

    return model

