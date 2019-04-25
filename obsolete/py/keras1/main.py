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

#input_shape = (32, 32, 3)
#input_shape = (64, 64, 3)
input_shape = (128, 128, 3)

def loadDataLabelsByTargetFile(path2Images):
    imgs = []
    targets = []
    with open(os.path.join(path2Images, 'targets.txt')) as f:
        for s in f:
            if len(s) < 2:
                continue
            tokens = s.split(" ")
            name = (tokens[0])
            imgs.append(cv2.imread(os.path.join(path2Images, name + '.png')))
            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])
            targets.append([x, y, z])
    imgs = np.array(imgs, dtype=np.float32)
    imgs = imgs.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    imgs /= 255.0
    targets = np.array(targets, dtype=np.float32)

    return imgs, targets

if __name__ == '__main__':
    img_dir = '/home/daiver/dump/'
<<<<<<< HEAD
    #input_shape = (128, 128, 1)
    input_shape = (64, 64, 1)
    #input_shape = (32, 32, 1)
    names_train = [ img_dir + "train/" + ("res_%d.png" % i) for i in xrange(71) ]
    imgs_train = np.array([cv2.imread(name, 0) for name in names_train], dtype=np.float32)
    imgs_train = imgs_train.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    imgs_train /= 255.0
    #y_train = [5.0 * i for i in xrange(71)]
    y_train = [5.0 * i for i in xrange(36)]
    y_train += [180 - 5.0 * i for i in xrange(0, 35)]
    #y_train /= 360.0

    names_test = [ img_dir + "test/" + ("res_%d.png" % i) for i in xrange(71) ]
    imgs_test = np.array([cv2.imread(name, 0) for name in names_test], dtype=np.float32)
    imgs_test = imgs_test.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    imgs_test /= 255.0
    #y_test = [2 + 5.0 * i for i in xrange(71)]
    y_test = [5.0 * i + 2 for i in xrange(36)]
    y_test += [180 - 5.0 * i - 2 for i in xrange(0, 35)]
    #y_test /= 360.0

    model = Sequential()
    model.add(Conv2D(3, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    #model.add(Conv2D(5, kernel_size=(3, 3), activation='relu'))
=======
    #input_shape = (16, 16, 3)

    train_dir = img_dir + "train/"
    imgs_train, y_train = loadDataLabelsByTargetFile(train_dir)
    #imgs_train, y_train = loadDataLabelsByFileName(train_dir)

    test_dir = img_dir + "test/"
    #imgs_test, y_test = loadDataLabelsByFileName(test_dir)
    imgs_test, y_test = loadDataLabelsByTargetFile(test_dir)

    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', use_bias=True, padding='same', input_shape=input_shape))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
    
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

    model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', use_bias=False, padding='same'))
    model.add(keras.layers.normalization.BatchNormalization())

    #model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

    #model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', use_bias=False, padding='same'))
    #model.add(keras.layers.normalization.BatchNormalization())

    model.add(AveragePooling2D(pool_size=(3, 3), strides=None, padding='same', data_format=None))

>>>>>>> 5be18515404d5f84709be2d9bf58ae103e3723ef
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss='mean_absolute_error', optimizer='adam')

<<<<<<< HEAD
    batch_size = 16
    epochs = 3000
=======
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
>>>>>>> 5be18515404d5f84709be2d9bf58ae103e3723ef
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

