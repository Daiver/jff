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
    img_dir = '/home/daiver/coding/jff/cpp/build-Raster-Desktop_Qt_5_7_0_GCC_64bit-Debug/'
    names = [
                img_dir + 'res_0.png',
                img_dir + 'res_1.png',
                img_dir + 'res_2.png',
                img_dir + 'res_3.png',
                img_dir + 'res_4.png',
                img_dir + 'res_5.png',
                img_dir + 'res_6.png',
                img_dir + 'res_7.png',
                img_dir + 'res_8.png',
                img_dir + 'res_9.png',
                img_dir + 'res_10.png',
                img_dir + 'res_11.png',
                img_dir + 'res_12.png',
                img_dir + 'res_13.png',
                img_dir + 'res_14.png',
                img_dir + 'res_15.png',
                img_dir + 'res_16.png',
                img_dir + 'res_17.png',
            ]
    imgs = np.array([cv2.imread(name, 0) for name in names], dtype=np.float32).reshape(-1, 32, 32, 1)
    imgs /= 255.0
    #imgs = np.vstack(imgs)
    print K.image_data_format(), imgs.shape
    input_shape = (32, 32, 1)
    num_classes = 18
    y_train = range(0, 18)
    #y_train = keras.utils.to_categorical(y_train, num_classes)

    model = Sequential()
    model.add(Conv2D(5, kernel_size=(3, 3),
		     activation='relu',
		     input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss=keras.losses.categorical_crossentropy,
    #              optimizer=keras.optimizers.Adadelta(),
    #              metrics=['accuracy'])

    batch_size = 18
    epochs = 1000
    model.fit(imgs, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(imgs, y_train))
    score = model.evaluate(imgs, y_train, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    for img in imgs:
        print np.argmax(model.predict(img.reshape(1, 32, 32, 1)))
        cv2.imshow('', img)
        cv2.waitKey()
