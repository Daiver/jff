import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import numpy as np
np.random.seed(42)
import cv2
import replace_geometry
from input_shape import input_shape

import keras
from keras import backend as K

if __name__ == '__main__':

    path2Model = sys.argv[1]
    path2Img   = sys.argv[2]
    path2Out   = sys.argv[3]
    model = keras.models.load_model(path2Model)
    img = cv2.imread(path2Img)
    path2Neutral = '/home/daiver/R3DS/Data/Render2ShapeRegression/534_Neutral_3mln_T_Wrapped2_OnlyFace.OBJ'
    neutralShapeLines = replace_geometry.readObj2Lines(path2Neutral)
    img = img.astype(np.float32).reshape(1, input_shape[0], input_shape[1], input_shape[2])
    img /= 255.0
    prediction = model.predict(img).reshape(-1)
    replace_geometry.replaceGeometryAndWrite(neutralShapeLines, prediction.reshape((-1, 3)), path2Out)

