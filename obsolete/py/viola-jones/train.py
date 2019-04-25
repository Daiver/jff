import numpy as np
import cv2
import os
import sys
import pickle

import adaboost

def readGrayImagesFromDir(dirName, size):
    filesNames = os.listdir(dirName)
    res = []
    for name in filesNames:
        img = cv2.imread(os.path.join(dirName, name), 0)
        if size[0] != -1:
            img = cv2.resize(img, size)
        res.append(img)

    return res

if __name__ == '__main__':
    negDir = sys.argv[1]
    posDir = sys.argv[2]

    size = (50, 50)
    negImgs = readGrayImagesFromDir(negDir, size)
    posImgs = readGrayImagesFromDir(posDir, size)
    negLabels = [0 for x in negImgs]
    posLabels = [1 for x in posImgs]
    labels = np.array(negLabels + posLabels)
    imgs = np.array(negImgs + posImgs)
    intergralImages = np.array([cv2.integral(x) for x in imgs])

    nTest = 1000
    indices = np.random.permutation(len(intergralImages))
    test_idx, training_idx = indices[:nTest], indices[nTest:]


    print 'training_idx', len(training_idx)
    print 'test_idx', len(test_idx)
#    coordSteps = [0.0,
                  #0.1, 
                  #0.2, 
                  #0.3, 
                  #0.4, 
                  #0.5, 
                  #0.6, 
                  #0.7, 
                  #0.8, 
                  #0.9]    
    coordSteps = [0.0,
                  0.2, 
                  0.4, 
                  0.5, 
                  0.6, 
                  0.8]

    scaleSteps = [
                  0.2, 
                  0.4, 
                  0.5, 
                  0.6, 
                  0.8, 
                  1.0]

    print 'Start learning'
    clfs, alphas = adaboost.trainAdaBoostClassifier(
            intergralImages[training_idx], labels[training_idx], coordSteps, scaleSteps, 30)

    pickle.dump(
            {
                'clfs' : clfs,
                'alphas' : alphas,
                'size' : size
                }
            , open('vj_classifier5.dump', 'w'))

    nErr = 0
    for i, (img, intImg, label) in enumerate(zip(imgs[test_idx], 
                                                 intergralImages[test_idx], 
                                                 labels[test_idx])):
        ans = adaboost.predict(clfs, alphas, intImg)
        if ans != label:
            nErr += 1
        print i, ans, label

    print nErr, nTest, float(nErr)/(nTest)
