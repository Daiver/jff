import numpy as np
import matplotlib.pyplot as plt

import keras

from keras.models import Sequential
from keras.layers import Dense, Input
np.random.seed(1337)

def mkModel():
    model = Sequential()
    model.add(Dense(1, input_shape=(1,)))
    return model

def trainAndShowDirect():
    model = mkModel()

    optimizer = keras.optimizers.SGD(lr=0.1)
    model.compile(optimizer=optimizer, loss='mse')

    line = lambda x: x * 0.5 - 6.0
    xs = np.random.uniform(-5, 5, 50)
    #xs = np.linspace(-5, 5, 50)
    ys = line(xs)

    model.fit(xs, ys, epochs=10, verbose=1)
    ys2 = model.predict(xs)

    plt.plot(xs, ys2, 'rx')

    xs = np.linspace(-5, 5, 50)
    ys = line(xs)
    plt.plot(xs, ys, 'go')
    plt.show()


def generator(line):
    while True:
        print 'gen called'
        xs = np.random.uniform(-5, 5, 50)
        ys = line(xs)
        yield (xs, ys)

def trainAndShowGen():
    model = mkModel()

    optimizer = keras.optimizers.SGD(lr=0.1)
    model.compile(optimizer=optimizer, loss='mse')

    line = lambda x: x * 0.5 - 6.0
    xs = np.linspace(-5, 5, 50)
    ys = line(xs)

    model.fit_generator(generator(line), steps_per_epoch=2, epochs=10, verbose=1, max_queue_size=2)
    ys2 = model.predict(xs)

    plt.plot(xs, ys, 'go')
    plt.plot(xs, ys2, 'rx')
    plt.show()

if __name__ == '__main__':
    trainAndShowGen()
    #trainAndShowDirect()
    


