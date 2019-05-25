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

class Quantizator:
    def __init__(self, nBins):
        self.nBins = nBins

    def fit(self, X):
        assert len(X.shape) == 1 or X.shape[1] == 1
        self.min = np.min(X)
        self.max = np.max(X)
        self.delta = self.max - self.min
        print self.min, self.max

    def transform(self, X):
        res = np.zeros((X.shape[0], self.nBins), dtype=np.float32)
        for i, x in enumerate(X):
            normalized = (x - self.min) / self.delta
            index = int(np.round(normalized * (self.nBins - 1)))
            res[i, index] = 1.0
        return res

    def inverse_transform(self, X):
        res = np.zeros(X.shape[0], dtype=np.float32)
        for i, x in enumerate(X):
            index = np.argmax(x)
            unnormalized = float(index) / (self.nBins - 1) * self.delta + self.min
            res[i] = unnormalized
        return res

def eval(model, transformer, xs, ys):
    res = 0.0
    predicted = model.predict(xs)
    predictedInv = transformer.inverse_transform(predicted)
    for r, t in zip(predictedInv, ys):
        res += (r - t)**2
    return res / xs.shape[0]

if __name__ == '__main__':
    xs = np.linspace(0, 1, 500)
    ys = np.array([(1.5 * x - 1)**2 for x in xs])

    nBins = 32
    transformer = Quantizator(nBins)
    transformer.fit(ys)
    ys2 = transformer.transform(ys)

    model = Sequential()
    model.add(Dense(10, input_shape=[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(nBins))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
                  loss='categorical_crossentropy',
                  #loss='mean_squared_error',
                  #optimizer=sgd,
                  optimizer='adam',
                  metrics=['accuracy']
                  )

    model.fit(xs, ys2,
          epochs=200,
          batch_size=32)
    print '>', model.evaluate(xs, ys2, verbose=False)
    print model.predict(xs[:10])
    print ys[:10]
    print eval(model, transformer, xs, ys)

