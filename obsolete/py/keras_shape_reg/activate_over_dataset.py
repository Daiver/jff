import os
if __name__ == '__main__':
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

    pass

import sys
import numpy as np
np.random.seed(42)
import cv2
import replace_geometry
#from input_shape import input_shape
from load_data import loadDataLabelsByTargetFile, loadDataMultiViewByTargetFile

import keras
from keras import backend as K

def evaluateModel(model, imgs, targets, loss='l2', postprocess=None):
    res = 0.0
    lossFunc = None
    if loss == 'l2':
        lossFunc = lambda x: np.sum(np.square(x)) / len(x)
    elif loss == 'l1':
        lossFunc = lambda x: np.sum(np.abs(x)) / len(x)
    else:
        assert(False)
    
    for img, target in zip(imgs, targets):
        img_reshaped = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        prediction = model.predict(img_reshaped).reshape(-1)
        if postprocess != None:
            prediction = postprocess(prediction)
        res += lossFunc(prediction - target)
    return res / len(imgs)


def runModelOverDataset(model, imgs, names, neutralShapeLines, destDir, postprocess=None):
    #destDir = "/home/daiver/results/train/"
    for img, name in zip(imgs, names):
        img_reshaped = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        prediction = model.predict(img_reshaped).reshape(-1)
        if postprocess != None:
            prediction = postprocess(prediction)
        replace_geometry.replaceGeometryAndWrite(
                neutralShapeLines, prediction.reshape((-1, 3)), destDir + "%s.obj" % str(name))


if __name__ == '__main__':
    import pickle
    path2Model = sys.argv[1]
    usePostprocessing = len(sys.argv) > 2
    if usePostprocessing:
        path2Scaler = sys.argv[2]
        with open(path2Scaler) as f:
            scaler = pickle.load(f)
        postprocess = lambda x: scaler.inverse_transform(x)
    else:
        postprocess = None

    #img_dir = '/home/daiver/R3DS/Data/Render2ShapeRegression/blendshape_data/'
    img_dir = '/home/daiver/R3DS/Data/Render2ShapeRegression/NeutralFacesDataset/'

    model = keras.models.load_model(path2Model)

    from keras.applications.xception import preprocess_input
    preprocess = lambda x: preprocess_input(x.astype(np.float32).reshape(-1, x.shape[0], x.shape[1], x.shape[2]))[0]
    preprocess = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).reshape((x.shape[0], x.shape[1], 1))


    train_dir = img_dir + "train/"
    imgs_train, y_train, names_train = loadDataMultiViewByTargetFile(train_dir, preprocess)

    test_dir = img_dir + "test/"
    imgs_test, y_test, names_test = loadDataMultiViewByTargetFile(test_dir, preprocess)

    if usePostprocessing:
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)

    score_train = model.evaluate(imgs_train, y_train, verbose=0)
    score_test  = model.evaluate(imgs_test, y_test, verbose=0)
    print('Train loss:', score_train)
    print('Test  loss:', score_test)
    print evaluateModel(model, imgs_train, y_train, 'l1')
    print evaluateModel(model, imgs_test, y_test, 'l1')



    #path2Neutral = '/home/daiver/R3DS/Data/Render2ShapeRegression/NeutralFacesCutted/AlexWrapped_Alligned.obj'
    path2Neutral = '/home/daiver/R3DS/Data/Render2ShapeRegression/NeutralFacesDataset/train/train_0.obj'
    neutralShapeLines = replace_geometry.readObj2Lines(path2Neutral)

    resultsDir = "/home/daiver/R3DS/Data/Render2ShapeRegression/results/"
    destTrainDir = resultsDir + "train/"
    destTestDir = resultsDir + "test/"

    runModelOverDataset(
            model, imgs_train, names_train, neutralShapeLines, destTrainDir, postprocess)
    runModelOverDataset(
            model, imgs_test, names_test, neutralShapeLines, destTestDir, postprocess)


