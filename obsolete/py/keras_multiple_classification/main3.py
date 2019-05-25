import numpy as np
np.random.seed(42)
import random
random.seed(42)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
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
        residuals = np.zeros(X.shape, dtype=np.float32)
        for i, x in enumerate(X):
            normalized = (x - self.min) / self.delta
            index = int(np.round(normalized * (self.nBins - 1)))
            unnormalized = float(index) / (self.nBins - 1) * self.delta + self.min
            residual = x - unnormalized
            res[i, index] = 1.0
            residuals[i] = residual
        return res, residuals

    def inverse_transform(self, X, residuals):
        res = np.zeros(X.shape[0], dtype=np.float32)
        for i, (x, r) in enumerate(zip(X, residuals)):
            index = np.argmax(x)
            unnormalized = float(index) / (self.nBins - 1) * self.delta + self.min
            unnormalized += r
            res[i] = unnormalized
        return res

def eval(model, transformer, xs, ys):
    res = 0.0
    predicted1, predicted2 = model.predict(xs)
    predictedInv = transformer.inverse_transform(predicted1, predicted2)
    for r, t in zip(predictedInv, ys):
        res += (r - t)**2
    return res / xs.shape[0]

if __name__ == '__main__':
    #xs = np.linspace(0, 1, 50)
    xs = np.linspace(0, 1, 500)
    ys = np.array([(1.5 * x - 1)**2 for x in xs])

    nBins = 32
    transformer = Quantizator(nBins)
    transformer.fit(ys)
    ys2, ys2_r = transformer.transform(ys)

    '''ys3 = transformer.inverse_transform(ys2, ys2_r)
    print ys
    print ys3
    print np.sum(np.square(ys - ys3))
    exit(0)'''

    inp = Input([1])
    x = inp
    x = Dense(10)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(10)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(10)(x)
    x = Activation('relu')(x)
    s = Dense(nBins)(x)
    s = Activation('softmax')(s)
    r = Dense(1)(x)
    model = Model(inputs=inp, outputs=[s, r])

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
                  loss=['categorical_crossentropy', 'mean_squared_error'],
                  loss_weights=[1.0, 2.0],
                  #loss='mean_squared_error',
                  #optimizer=sgd,
                  optimizer='adam',
                  #metrics=['accuracy']
                  )

    model.fit(xs, [ys2, ys2_r],
          epochs=200,
          batch_size=32)
    #print '>', model.evaluate(xs, ys2, verbose=False)
    #print model.predict(xs[:10])
    print eval(model, transformer, xs, ys)

