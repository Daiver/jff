import numpy as np
import cv2
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K

if __name__ == '__main__':
    #img_dir = '/home/daiver/coding/jff/cpp/build-Raster-Desktop_Qt_5_7_0_GCC_64bit-Debug/'
    img_dir = '/home/daiver/dump/'
    input_shape = (128, 128, 1)
    #input_shape = (64, 64, 1)
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
    model.add(Conv2D(5, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    #model.add(Conv2D(5, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    #model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss='mean_absolute_error', optimizer='adam')

    batch_size = 16
    epochs = 2000
    model.fit(imgs_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(imgs_test, y_test))
    score = model.evaluate(imgs_test, y_test, verbose=0)
    print('Test loss:', score)
    #exit(0)
    #print('Test accuracy:', score[1])
    for i, img in enumerate(imgs_test):
        print (model.predict(img.reshape(1, input_shape[0], input_shape[1], input_shape[2]))) , y_test[i]
        cv2.imshow('', img)
        cv2.waitKey()
