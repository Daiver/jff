import numpy as np
np.random.seed(42)
import random
random.seed(42)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

if __name__ == '__main__':
    xs = np.linspace(0, 1, 500)
    ys = np.array([(1.5 * x - 1)**2 for x in xs])

    model = Sequential()
    model.add(Dense(10, input_shape=[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(10))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
                  #loss='categorical_crossentropy',
                  loss='mean_squared_error',
                  #optimizer=sgd,
                  optimizer='adam',
                  #metrics=['accuracy']
                  )

    model.fit(xs, ys,
          epochs=200,
          batch_size=32)
    print '>', model.evaluate(xs, ys, verbose=False)
    print model.predict(xs[:10])
    print ys[:10]

