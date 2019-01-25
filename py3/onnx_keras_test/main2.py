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
    kerasModel = keras.applications.ResNet50()
    onnxModel = onnxkeras.keras_model_to_onnx_model(kerasModel)
    onnxkeras.save(onnxModel, 'keras_resnet.proto')
