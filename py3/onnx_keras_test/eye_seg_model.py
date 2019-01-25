import sys
PATH_TO_ONNX_KERAS = '/home/daiver/coding/onnx-keras'
sys.path.append(PATH_TO_ONNX_KERAS)
sys.path.append('/home/daiver/R3DS/Kirill/FaceSegmentation1/eye_seg1/')
import frontend as onnxkeras
import keras
import onnx
import tensorflow as tf

from keras.models import Model
from keras.utils.generic_utils import CustomObjectScope
from losses import *
import linknet1

def loadKerasModel(modelName):
    print ('reading...', modelName)
    with CustomObjectScope({
        'relu6': keras.applications.mobilenet.relu6,
        'f' : weightedBinaryCrossentropy([1, 1]),
        'bceSoftDice' : bceSoftDice,
        }):
        model = keras.models.load_model(modelName)
        inp = model.inputs[0]
        targetShape = inp.shape[1:3]
        out = model.outputs[-1]
        model = Model(inputs=[inp], outputs=[out])
    return model

from losses import *
from focal_loss import *


if __name__ == '__main__':
    kerasModelName = '/home/daiver/R3DS/Kirill/FaceSegmentation1/eye_seg1/good_models_leye/ex351_checkpoint.h5'
    #kerasModel = loadKerasModel((kerasModelName))
    kerasModel = linknet1.LinkNet()
    onnxModel = onnxkeras.keras_model_to_onnx_model(kerasModel)
    onnxkeras.save(onnxModel, 'eye_seg.proto')

