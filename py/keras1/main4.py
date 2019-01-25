import datetime
import os
import numpy as np
np.random.seed(42)
import cv2

import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K

input_shape = (32, 32, 3)

def loadDataLabelsByFileName(path2Images):
    names = [os.path.join(path2Images, name) for name in os.listdir(path2Images) ]
    imgs = np.array([cv2.imread(name) for name in names], dtype=np.float32)
    imgs = imgs.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    imgs /= 255.0

    y = [float(os.path.splitext(os.path.basename(name))[0]) for name in names]
 
    return imgs, y

def loadDataLabelsByTargetFile(path2Images):
    names = [
            os.path.join(path2Images, name) 
            for name in os.listdir(path2Images) 
            if os.path.splitext(name)[1] != ".txt"
        ]

    nameIndex2Index = {}
    for i, name in enumerate(names):
        nameIndex = (os.path.splitext(os.path.basename(name))[0])
        nameIndex2Index[nameIndex] = i

    imgs = np.array([cv2.imread(name) for name in names], dtype=np.float32)
    imgs = imgs.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    imgs /= 255.0
    targets = np.zeros((len(imgs), 3))
    #targets = np.zeros((len(imgs), 2))
    #targets = np.zeros((len(imgs)))
    with open(os.path.join(path2Images, 'targets.txt')) as f:
        for s in f:
            if len(s) in [0, 1]:
                continue
            tokens = s.split(" ")
            nameIndex = (tokens[0])
            index = nameIndex2Index[nameIndex]
            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])
            #targets[index] = x
            #targets[index, :] = [x, y]
            targets[index, :] = [x, y, z]

    return imgs, targets

if __name__ == '__main__':
    #img_dir = '/home/daiver/coding/jff/cpp/build-Raster-Desktop_Qt_5_7_0_GCC_64bit-Debug/'
    img_dir = '/home/daiver/dump/'
    #input_shape = (128, 128, 3)
    #input_shape = (64, 64, 3)
    #input_shape = (16, 16, 3)

    train_dir = img_dir + "train/"
    imgs_train, y_train = loadDataLabelsByTargetFile(train_dir)
    #imgs_train, y_train = loadDataLabelsByFileName(train_dir)

    test_dir = img_dir + "test/"
    #imgs_test, y_test = loadDataLabelsByFileName(test_dir)
    imgs_test, y_test = loadDataLabelsByTargetFile(test_dir)

    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', use_bias=False, padding='same', input_shape=input_shape))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', use_bias=False, padding='same'))
    model.add(keras.layers.normalization.BatchNormalization())
    #model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', use_bias=False, padding='same'))
    #model.add(keras.layers.normalization.BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', use_bias=False, padding='same'))
    model.add(keras.layers.normalization.BatchNormalization())
    #model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', use_bias=False, padding='same'))
    #model.add(keras.layers.normalization.BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', use_bias=False, padding='same'))
    model.add(keras.layers.normalization.BatchNormalization())

    #model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    #model.add(Conv2D(24, kernel_size=(3, 3), activation='relu'))
    #model.add(keras.layers.normalization.BatchNormalization())
    #model.add(Conv2D(6, kernel_size=(3, 3), activation='relu'))
    #model.add(AveragePooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=None, padding='same', data_format=None))
    #model.add(AveragePooling2D(pool_size=(10, 10), strides=None, padding='valid', data_format=None))
    #model.add(MaxPooling2D(pool_size=(5, 5), strides=None, padding='valid', data_format=None))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss='mean_absolute_error', optimizer='adam')

    #batch_size = 8
    #batch_size = 32
    batch_size = 64
    #epochs = 100
    epochs = 500
    #epochs = 1000
    #epochs = 3000
    #epochs = 5000
    #epochs = 10000
    #epochs = 20000
    #epochs = 80000
    model.fit(imgs_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=True,
              validation_data=(imgs_test, y_test),
              callbacks = [keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)])
    score_train = model.evaluate(imgs_train, y_train, verbose=0)
    score_test  = model.evaluate(imgs_test, y_test, verbose=0)
    print('Train loss:', score_train)
    print('Test  loss:', score_test)
    model.save("%s_train_l_%s_test_l_%s.h5" % (str(datetime.datetime.now()), str(score_train), str(score_test)))

    #exit(0)
    #print('Test accuracy:', score[1])
    
    for i, img in enumerate(imgs_test):
        print (model.predict(img.reshape(1, input_shape[0], input_shape[1], input_shape[2]))) , y_test[i]
        cv2.imshow('', img)
        cv2.waitKey()

