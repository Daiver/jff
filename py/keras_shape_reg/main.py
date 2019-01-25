import datetime
import os
import pickle
import math
import numpy as np
np.random.seed(42)
from sklearn import preprocessing
import cv2
from load_data import loadDataLabelsByTargetFile, loadDataMultiViewByTargetFile
from input_shape import input_shape
from lrdecays import (
        expDecay, invSqrtWithLastStepsDecay, invSqrtDecay, 
        linearDecay, linearWithRampUp,
        sqrtDecay
    )
import replace_geometry

import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K

import activate_over_dataset

from model import mkDarknetLight #, mkPretrainedVGG16Model
#from model import mkSharedByViewsModel, mkCrazyCascade
from res_model import mkResnet

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

if __name__ == '__main__':
    #img_dir = '/home/daiver/blendshapes_dump/'
    #img_dir = '/home/daiver/R3DS/Data/Render2ShapeRegression/blendshape_data/'
    home_dir = '/home/daiver/'
    #home_dir = '/home/r3ds/'
    img_dir = home_dir + 'R3DS/Data/Render2ShapeRegression/NeutralFacesDataset/'

    preprocess = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).reshape((x.shape[0], x.shape[1], 1))

    train_dir = img_dir + "train/"
    imgs_train, y_train, names_train = loadDataMultiViewByTargetFile(train_dir, preprocess)
    print imgs_train.shape


    test_dir = img_dir + "test/"
    imgs_test, y_test, names_test = loadDataMultiViewByTargetFile(test_dir, preprocess)

    curTime = datetime.datetime.now()

    outputScaler = preprocessing.StandardScaler()
    outputScaler.fit(y_train)
    y_train = outputScaler.transform(y_train)
    y_test = outputScaler.transform(y_test)

    if outputScaler != None:
        with open('scalers/%s.data' % str(curTime), 'w') as f:
            pickle.dump(outputScaler, f)

    nFeats = len(y_train[0])

    model = mkDarknetLight(nFeats, input_shape)
    #model = mkResnet(nFeats, input_shape)

    print "nParams", model.count_params()
    #model.summary()
    initialLearningRate = 0.001
    #initialLearningRate = 0.002
    #optimizer = optimizers.Adam(decay=1.0/100)
    optimizer = optimizers.Adam(lr=initialLearningRate)
    #optimizer = optimizers.SGD(lr=1.1, decay=0e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    #model.compile(loss='mean_absolute_error', optimizer=optimizer)

    #batch_size = 32
    #batch_size = 64
    batch_size = 128
    #batch_size = 256

    #epochs = 10
    #epochs = 200
    #epochs = 500
    epochs = 2000
    #epochs = 5000

    print 'Start coarse training'
    #model.fit(imgs_train, y_train, batch_size=batch_size, epochs=20, verbose=False)
    model.fit(imgs_train, y_train, 
            batch_size=8, epochs=20, verbose=True, validation_split=0.1)


    name4checkpoint = "checkpoints/%s_ep_{epoch:02d}_train_l_{loss:.5f}_test_l_{val_loss:.5f}.h5" % (str(curTime))
    
    #decayFunc = expDecay(initial_lrate=initialLearningRate, epochs_drop=30)
    decayFunc = linearDecay(initial_lrate=initialLearningRate, nIters=epochs)
    #decayFunc = sqrtDecay(initial_lrate=initialLearningRate, nIters=epochs)
    lrate = LearningRateScheduler(decayFunc)
    reduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                          patience=10, 
                          min_lr=0.00001, verbose=True)
    callbacks = [
                keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True),
                keras.callbacks.ModelCheckpoint(
                    name4checkpoint, 
                    monitor='val_loss', verbose=True, 
                    save_weights_only=False, 
                    save_best_only=True,
                    mode='auto', period=50),
                lrate,
                #reduceLr,
            ]
    print 'Start training'
    model.fit(imgs_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=True,
              #validation_data=(imgs_test[:128], y_test[:128]),
              validation_data=(imgs_test, y_test),
              callbacks = callbacks)

    score_train = model.evaluate(imgs_train, y_train, verbose=0)
    score_test  = model.evaluate(imgs_test, y_test, verbose=0)
    print('Train loss:', score_train)
    print('Test  loss:', score_test)
    print activate_over_dataset.evaluateModel(model, imgs_train, y_train, 'l2')
    print activate_over_dataset.evaluateModel(model, imgs_test, y_test, 'l2')
    name4finalmodel = "models/%s_train_l_%s_test_l_%s.h5" % (str(datetime.datetime.now()), str(score_train), str(score_test))
    model.save(name4finalmodel)

    path2Neutral = home_dir + 'R3DS/Data/Render2ShapeRegression/NeutralFacesDataset/train/train_0.obj'

    neutralShapeLines = replace_geometry.readObj2Lines(path2Neutral)

    resultsDir = home_dir + "R3DS/Data/Render2ShapeRegression/results/"
    destTrainDir = resultsDir + "train/"
    destTestDir = resultsDir + "test/"

    #postprocess = None
    postprocess = lambda x: outputScaler.inverse_transform(x)
    activate_over_dataset.runModelOverDataset(
            model, imgs_train, names_train, neutralShapeLines, destTrainDir, postprocess)
    activate_over_dataset.runModelOverDataset(
            model, imgs_test, names_test, neutralShapeLines, destTestDir, postprocess)

