import random
random.seed(42)
import numpy as np
np.random.seed(42)
import cv2
import os
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import decode_predictions
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import Input, Activation, Concatenate
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D,UpSampling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Lambda, Add, add, concatenate


def convBnAct(nFilters, kernel_size=(3, 3), activation='relu', name=None):
    initializer = 'he_normal'
    def f(x):
        cName = None if name == None else name + '_conv'
        conv1 = Conv2D(nFilters, kernel_size=kernel_size, 
                kernel_initializer=initializer, padding='same',
                name=cName)(x)
        bName = None if name == None else name + '_bn'
        bn1   = BatchNormalization(name=bName)(conv1)
        aName = None if name == None else name + '_act'
        act1  = Activation(activation, name=aName)(bn1)
        return act1
    return f

###WARNING: In some implementations peoples adds additional Conv1x1 to make nFilters match
def residualBlock(nExternalFilters, nInnerFilters=None):
    if nInnerFilters == None:
        nInnerFilters = nExternalFilters // 2
    def f(xInp):
        x = xInp
        x = convBnAct(nInnerFilters   , (1, 1))(x)
        x = convBnAct(nInnerFilters   , (3, 3))(x)
        x = convBnAct(nExternalFilters, (3, 3))(x)
        x = add([xInp, x])
        return x
    return f

def hourglassBlockv1(nExternalFilters, nInnerFilters=None, nLvls=4):
    def f(x):
        upBrnch = residualBlock(nExternalFilters, nInnerFilters)(x)
        lowDwnSmpl = MaxPooling2D()(x)
        lowRes1    = residualBlock(nExternalFilters, nInnerFilters)(lowDwnSmpl)
        if nLvls == 1:
            lowRes2 = residualBlock(nExternalFilters, nInnerFilters)(lowRes1)
        else:
            lowRes2 = hourglassBlockv1(nExternalFilters, nInnerFilters, nLvls - 1)(lowRes1)
        lowRes3    = residualBlock(nExternalFilters, nInnerFilters)(lowRes2)
        lowUpSmpl  = UpSampling2D()(lowRes3)
        res = add([lowUpSmpl, upBrnch])
        return res
    return f

def hourglassBlockv2(nExternalFilters, nInnerFilters=None, nLvls=4):
    assert False # Not properly implemented
    def f(x):
        upBrnch = residualBlock(nExternalFilters, nInnerFilters)(x)
        lowDwnSmpl = MaxPooling2D()(x)
        lowRes1    = residualBlock(nExternalFilters, nInnerFilters)(lowDwnSmpl)
        if nLvls == 1:
            lowRes2 = residualBlock(nExternalFilters, nInnerFilters)(lowRes1)
        else:
            lowRes2 = hourglassBlockv2(2*nExternalFilters, 2*nInnerFilters, nLvls - 1)(lowRes1)
        lowRes3    = residualBlock(nExternalFilters, nInnerFilters)(lowRes2)
        lowUpSmpl  = UpSampling2D()(lowRes3)
        res = add([lowUpSmpl, upBrnch])
        return res
    return f

def mkHourglass(targetShape, nFilters=16, nInpChannels=3, nFinalOutputs=1):
    inputShape = (targetShape[0], targetShape[1], nInpChannels)
    dropRate = 0.2
    outputs = []

    inp = Input(inputShape)
    x = inp

    x = convBnAct(nFilters, (3, 3))(x)

    x = hourglassBlockv1(nFilters, nFilters // 2, 4)(x)

    enableSecondHourglass = False
    if enableSecondHourglass:
        x            = convBnAct(nFilters,(1, 1))(x)
        drop1        = Dropout(dropRate)(x)
        out1         = convBnAct(nFinalOutputs, (1, 1), 'sigmoid', name='out1')(drop1)
        outputs.append(out1)
        out1Reshaped = convBnAct(nFilters, (1, 1))(out1)
        x = convBnAct(nFilters,(1, 1))(x)
        x = add([x, out1Reshaped])

        x = hourglassBlockv1(nFilters, nFilters // 2, 4)(x)

    #x            = convBnAct(nFilters,(1, 1))(x)
    #drop2        = Dropout(dropRate)(x)
    #out2         = convBnAct(nFinalOutputs, (1, 1), 'sigmoid', name='out2')(drop2)
    #out2Reshaped = convBnAct(nFilters, (1, 1))(out2)
    #x = convBnAct(nFilters,(1, 1))(x)
    #x = add([x, out2Reshaped])

    #x = hourglassBlockv1(nFilters, nFilters // 2, 4)(x)

    x = convBnAct(nFilters,(1, 1))(x)
    x = convBnAct(nFilters,(1, 1))(x)

    x = Dropout(dropRate)(x)
    x = Conv2D(nFinalOutputs, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid', name='out_final_act')(x)

    outputs.append(x)

    model = Model(inputs = [inp], outputs = outputs)

    #print model.summary()
    return model


