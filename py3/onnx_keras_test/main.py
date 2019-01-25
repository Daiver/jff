import sys
PATH_TO_ONNX_KERAS = '/home/daiver/coding/onnx-keras/'
sys.path.append(PATH_TO_ONNX_KERAS)
import frontend as onnxkeras
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import onnx
import numpy as np

def main1():
    kerasModel = keras.applications.ResNet50()
    #layer = Dense(1, input_shape=(1,))
    #layer.set_weights([[13]])
    #kerasModel = Sequential([
    #    layer
    #])
    #print('before get_weights')
    #print(layer.get_weights())
    #print('after get_weights')
    #layer.set_weights([np.array([[13]]), np.array([5])])
    #print('before get_weights')
    #print(layer.get_weights())
    #print('after get_weights')
    onnxModel = onnxkeras.keras_model_to_onnx_model(kerasModel)
    onnxkeras.save(onnxModel, 'resnet50.proto')

    #print(kerasModel.predict(np.array([[0.01]])))

if __name__ == '__main__':
    kerasModel = keras.applications.ResNet50()
    onnxModel = onnxkeras.keras_model_to_onnx_model(kerasModel)
    onnxkeras.save(onnxModel, 'resnet50.proto')

    #print(kerasModel.predict(np.array([[0.01]])))
