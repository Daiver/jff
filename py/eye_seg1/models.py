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


def mkModel():
    inputShape = (targetShape[0], targetShape[1], 3)
    model = Sequential()
    nFilters = 32

    model.add(Conv2D(nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        dilation_rate = 2,
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(nFilters, kernel_size=(3, 3), strides=(1, 1), 
        dilation_rate = 2,
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(nFilters, kernel_size=(3, 3), strides=(1, 1), 
        dilation_rate = 2,
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(nFilters, kernel_size=(3, 3), strides=(1, 1), 
        dilation_rate = 2,
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.2))
    model.add(Conv2D(1, kernel_size=(1, 1), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    return model

def mkModel2():
    inputShape = (targetShape[0], targetShape[1], 3)
    model = Sequential()
    nFilters = 32
    #nFilters = 128

    model.add(Conv2D(nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(nFilters, kernel_size=(3, 3), strides=(2, 2), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(2*nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(2*nFilters, kernel_size=(3, 3), strides=(2, 2), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(4*nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(4*nFilters, kernel_size=(3, 3), strides=(2, 2), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(8*nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(8*nFilters, kernel_size=(3, 3), strides=(2, 2), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(16*nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #DECODER
    model.add(UpSampling2D())
    model.add(Conv2D(8*nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(4*nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(2*nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(nFilters, kernel_size=(3, 3), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.2))
    model.add(Conv2D(1, kernel_size=(1, 1), strides=(1, 1), 
        kernel_initializer='he_normal',
        use_bias=True, padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    return model

def mkPretrainedMobileNet():
    model = keras.applications.mobilenet.MobileNet(input_shape=(targetShape[0], targetShape[1], 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None)
    
    #for l in model.layers:
    #    l.trainable = False

    input = model.inputs[0]
    output = model.outputs[0]
    x = output

    nFilters = 1024
    nFilters /= 2
    for i in xrange(5):
        x = UpSampling2D()(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(nFilters, kernel_size=(3, 3), padding='same', 
                kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(nFilters, kernel_size=(3, 3), padding='same', 
                kernel_initializer='he_normal')(x)
        nFilters /= 2

        if i == 0:
            oldL = model.get_layer('conv_pw_11_relu').output
            x = Add()([x, oldL])
        if i == 1:
            oldL = model.get_layer('conv_pw_5_relu').output
            x = Add()([x, oldL])
        if i == 2:
            oldL = model.get_layer('conv_pw_3_relu').output
            x = Add()([x, oldL])
        if i == 3:
            oldL = model.get_layer('conv_pw_1_relu').output
            x = Add()([x, oldL])

    #x = Dropout(0.2)(x)
    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), 
        kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs = [input], outputs=[x])
    print model.summary()
    return model

def mkPretrainedVGG16():
    model = keras.applications.VGG16(input_shape=(targetShape[0], targetShape[1], 3), include_top=False, weights='imagenet')
    
    #for l in model.layers:
    #    l.trainable = False

    input = model.inputs[0]
    vggOutput = model.outputs[0]
    x = vggOutput

    block4Pool = model.get_layer('block4_pool').output
    block4Shortcut = Conv2D(2, kernel_size=(1, 1), kernel_initializer='he_normal')(block4Pool)
    block3Pool = model.get_layer('block3_pool').output
    block3Shortcut = Conv2D(2, kernel_size=(1, 1), kernel_initializer='he_normal')(block3Pool)

    x = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), 
        kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(2, kernel_size=(4, 4), strides=(2, 2),
            kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, block4Shortcut])
    
    x = Conv2DTranspose(2, kernel_size=(4, 4), strides=(2, 2),
            kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, block3Shortcut])
    
    x = Conv2DTranspose(2, kernel_size=(16, 16), strides=(8, 8),
            kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    

    fcnOutput = x

    x = vggOutput
    x = Flatten()(x)
    x = Dense(4*4*512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((4, 4, 512))(x)
    
    x = Conv2DTranspose(512, kernel_size=(4, 4), strides=(1, 1), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)#8 8
    x = Conv2DTranspose(512, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(512, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(512, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)#16 16
    x = Conv2DTranspose(256, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)#32 32 
    x = Conv2DTranspose(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)#64 64
    x = Conv2DTranspose(64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)#128 128
    x = Conv2DTranspose(64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(2, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    deconvOutput = x

    x = Concatenate()([deconvOutput, fcnOutput])

    #x = Dropout(0.2)(x)
    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), 
        kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs = [input], outputs=[x])
    print model.summary()
    return model

def convBnAct(nFilters, kernel_size=(3, 3)):
    initializer = 'he_normal'
    def f(x):
        conv1 = Conv2D(nFilters, kernel_size=kernel_size, 
                kernel_initializer=initializer, padding='same')(x)
        bn1   = BatchNormalization()(conv1)
        act1  = Activation('relu')(bn1)
        return act1
    return f

def convBlock(nFilters):
    def f(xInput):
        skipCon = add
        x0 = convBnAct(nFilters, (1, 1))(xInput)
        #x0 = xInput
        x1 = convBnAct(nFilters)(x0)
        x2 = convBnAct(nFilters)(x1)

        #x = x2
        x  = skipCon([x0, x2])
        return x
    return f

def mkModel3(targetShape, nOutputs=1):
    inputShape = (targetShape[0], targetShape[1], 3)

    #skipCon = add
    skipCon = concatenate

    model = Sequential()
    #nFilters = 16
    nFilters = 32
    #nFilters = 64
    #nFilters = 128

    inp = Input(inputShape)
    x = inp

    #encoder
    x = convBlock(nFilters)(x)

    poolBlock1Out = x

    x = MaxPooling2D()(x)# / 2
    x = convBlock(2*nFilters)(x)

    poolBlock2Out = x

    x = MaxPooling2D()(x)# / 4
    x = convBlock(4*nFilters)(x)

    poolBlock3Out = x

    x = MaxPooling2D()(x)# / 8
    x = convBlock(8*nFilters)(x)

    poolBlock4Out = x

    x = MaxPooling2D()(x)# / 16
    x = convBlock(16*nFilters)(x)

    poolBlock5Out = x

    #decoder

    x = UpSampling2D()(x)# 8
    unpoolBlock4Out = x
    x = skipCon([unpoolBlock4Out, poolBlock4Out])

    x = convBlock(8*nFilters)(x)

    x = UpSampling2D()(x)# 4
    unpoolBlock3Out = x
    x = skipCon([unpoolBlock3Out, poolBlock3Out])

    x = convBlock(4*nFilters)(x)

    x = UpSampling2D()(x)# 2
    unpoolBlock2Out = x
    x = skipCon([unpoolBlock2Out, poolBlock2Out])

    x = convBlock(2*nFilters)(x)

    x = UpSampling2D()(x)# 1
    unpoolBlock1Out = x
    x = skipCon([unpoolBlock1Out, poolBlock1Out])

    x0 = convBlock(nFilters)(x)
    x = skipCon([x, x0])
    x0 = convBlock(nFilters)(x)
    x = skipCon([x, x0])

    x = Dropout(0.1)(x)
    x = Conv2D(nOutputs, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs = [inp], outputs = [x])

    print model.summary()
    return model

def convBlock(nFilters):
    def f(xInput):
        skipCon = add
        x0 = convBnAct(nFilters, (1, 1))(xInput)
        x1 = convBnAct(nFilters)(x0)
        x2 = convBnAct(nFilters)(x1)
        x  = skipCon([x0, x2])
        return x
    return f

def mkModel4(targetShape):
    inputShape = (targetShape[0], targetShape[1], 3)

    #skipCon = add
    skipCon = concatenate

    model = Sequential()
    #nFilters = 16
    nFilters = 32
    #nFilters = 128

    inp = Input(inputShape)
    x = inp

    #encoder
    x = convBlock(nFilters)(x)
    x = convBlock(nFilters)(x)

    poolBlock1Out = x

    x = MaxPooling2D()(x)# / 2
    x = convBlock(2*nFilters)(x)
    x = convBlock(2*nFilters)(x)
    x = convBlock(2*nFilters)(x)

    poolBlock2Out = x

    x = MaxPooling2D()(x)# / 4
    x = convBlock(4*nFilters)(x)
    x = convBlock(4*nFilters)(x)
    x = convBlock(4*nFilters)(x)

    poolBlock3Out = x

    x = MaxPooling2D()(x)# / 8
    x = convBlock(8*nFilters)(x)
    x = convBlock(8*nFilters)(x)
    x = convBlock(8*nFilters)(x)

    poolBlock4Out = x

    unp1 = Conv2D(nFilters, kernel_size=(1, 1), 
            padding='same', kernel_initializer='he_normal')(poolBlock1Out)
    unp1 = BatchNormalization()(unp1)
    unp1 = Activation('relu')(unp1)

    unp2 = Conv2D(nFilters, kernel_size=(1, 1), 
            padding='same', kernel_initializer='he_normal')(poolBlock2Out)
    unp2 = BatchNormalization()(unp2)
    unp2 = Activation('relu')(unp2)
    unp2 = UpSampling2D((2, 2))(unp2)

    unp3 = Conv2D(nFilters, kernel_size=(1, 1), 
            padding='same', kernel_initializer='he_normal')(poolBlock3Out)
    unp3 = BatchNormalization()(unp3)
    unp3 = Activation('relu')(unp3)
    unp3 = UpSampling2D((4, 4))(unp3)

    unp4 = Conv2D(nFilters, kernel_size=(1, 1), 
            padding='same', kernel_initializer='he_normal')(poolBlock4Out)
    unp4 = BatchNormalization()(unp4)
    unp4 = Activation('relu')(unp4)
    unp4 = UpSampling2D((8, 8))(unp4)

    x = concatenate([unp1, unp2, unp3, unp4])
    x = convBnAct(nFilters)(x)
    x = convBnAct(nFilters)(x)

    #x = Dropout(0.5)(x)
    x = Conv2D(1, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs = [inp], outputs = [x])

    print model.summary()
    return model

