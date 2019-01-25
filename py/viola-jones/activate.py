import numpy as np
import cv2
import sys
import pickle

import common
import adaboost

def getSlidingWindows(img, coordSteps, scale2DSteps):
     #integralImg.shape
    width, height = img.shape
    for coordStepX in coordSteps:
        for coordStepY in coordSteps:
            for scale2DStep in scale2DSteps:
                w, h = width * scale2DStep, height * scale2DStep
                x, y = coordStepX * width, coordStepY * height
                rect = [x, y, w, h]
                frame = common.cutPatch(img, rect)
                if x+w >= width or y+h >= height:
                    continue
                yield frame, rect

def detect(img, coordSteps, scale2DSteps, size, clfs, alphas):
    res = []
    for frame, rect in getSlidingWindows(img, coordSteps, scale2DSteps):
        if min(frame.shape) <= 0:
            continue
        #print frame.shape, frame.dtype, rect, size
        frameSized = cv2.resize(frame, size)
        frameInt = cv2.integral(frameSized)
        ans = adaboost.predict(clfs, alphas, frameInt)
        if ans == 1:
            res.append(rect)
    return res

if __name__ == '__main__':
    coordSteps   = np.linspace(0, 1.0, 10)
    scale2DSteps = np.linspace(0.1, 1.0, 5)
    classifier = pickle.load(open(sys.argv[1]))
    img = cv2.imread(sys.argv[2], 0)
    print classifier['size']
    res = detect(img, coordSteps, scale2DSteps, classifier['size'], classifier['clfs'], classifier['alphas'])
    print res
    print len(res)
    for rect in res:
        cv2.rectangle(img, 
                (int(rect[0]), int(rect[1])),
                (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
                255)
    #cv2.imwrite('res.png', img)
