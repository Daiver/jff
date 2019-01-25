import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import os
import random

import common
import config

def cutPatches(img):
    patchSize = (12, 12)
    step = 5
    res = []
    for i in xrange(0, img.shape[0], step):
        for j in xrange(0, img.shape[1], step):
            res.append(common.cutImage(img, (i, j, patchSize[0], patchSize[1])))
    return res


if __name__ == '__main__':
    dirs = map(lambda x: os.path.join(config.cropped_dataset_path, x), 
            os.listdir(config.cropped_dataset_path))

    dirs = dirs[:10]
    print dirs
    classes = []

    for d in dirs:
        paths = map(lambda x: os.path.join(d, x), os.listdir(d))
        classes.append(map(cv2.imread, paths))
        print len(classes[-1])

    print len(classes)
