from keras_train_mnist import *

import sacred
from sacred.stflow import LogFileWriter
from sacred.utils import apply_backspaces_and_linefeeds

ex = sacred.Experiment('mnist_keras')
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    epochs = 2


@ex.automain
@LogFileWriter(ex)
def main(epochs):
    res = trainMnist(epochs)
    return res
