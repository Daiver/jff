import sys
PATH_TO_ONNX_KERAS = '/home/daiver/coding/onnx-keras'
sys.path.append(PATH_TO_ONNX_KERAS)
import frontend as onnxkeras
import keras
import onnx
import tensorflow as tf

if __name__ == '__main__':
    print(tf.__version__)
    print(keras.__version__)
    kerasModel = keras.models.Sequential([
        keras.layers.Dense(10, input_shape=(10,)),
        keras.layers.BatchNormalization()
    ])
    onnxModel = onnxkeras.keras_model_to_onnx_model(kerasModel)
    onnxkeras.save(onnxModel, 'tmp.proto')
