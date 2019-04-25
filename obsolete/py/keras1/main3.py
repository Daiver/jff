import os
import numpy as np
np.random.seed(42)
import cv2

import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
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

if __name__ == '__main__':
    #img_dir = '/home/daiver/coding/jff/cpp/build-Raster-Desktop_Qt_5_7_0_GCC_64bit-Debug/'
    img_dir = '/home/daiver/dump/'
    #input_shape = (128, 128, 3)
    #input_shape = (64, 64, 3)
    #input_shape = (16, 16, 3)

    train_dir = img_dir + "train/"
    imgs_train, y_train = loadDataLabelsByFileName(train_dir)

    test_dir = img_dir + "test/"
    imgs_test, y_test = loadDataLabelsByFileName(test_dir)

    model = Sequential()
    model.add(Conv2D(3, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    #model.add(AveragePooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    #model.add(Conv2D(3, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 3)))
    #model.add(AveragePooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=None, padding='valid', data_format=None))
    #model.add(MaxPooling2D(pool_size=(5, 5), strides=None, padding='valid', data_format=None))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss='mean_absolute_error', optimizer='adam')

    batch_size = 8
    #epochs = 1000
    #epochs = 3000
    #epochs = 5000
    epochs = 10000
    #epochs = 20000
    #epochs = 80000
    model.fit(imgs_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=True,
              validation_data=(imgs_test, y_test),
              callbacks = [keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)])
    score = model.evaluate(imgs_test, y_test, verbose=0)
    print('Test loss:', score)
    exit(0)
    #print('Test accuracy:', score[1])
    
    for i, img in enumerate(imgs_test):
        print (model.predict(img.reshape(1, input_shape[0], input_shape[1], input_shape[2]))) , y_test[i]
        cv2.imshow('', img)
        cv2.waitKey()
