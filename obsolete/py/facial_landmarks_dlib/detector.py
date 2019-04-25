import dlib
from skimage import io
import numpy
import sys
import os

predictorPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_68_face_landmarks.dat")

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorPath)

def runDetector(img):
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        shape = [[i.x, i.y] for i in shape.parts()]
        shape = numpy.array(shape)
    return shape

def show(img, shape):
    import matplotlib.pyplot as plt
    #import cv2
    #cv2.imshow('', img)
    #cv2.waitKey()
    plt.imshow(img)
    plt.scatter(shape[:, 0], shape[:, 1], c='w', s=8)
    plt.show()

def writeShape(shape, outputLandmarksName):
    with open(outputLandmarksName, 'w') as f:
        for x in shape:
            f.write('%s %s\n' % (x[0], x[1]))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dlib facial landmarks detector')
    parser.add_argument('inputImgName')
    parser.add_argument('outputLandmarksName')
    parser.add_argument('--show', help='draw landmarks on screen', action='store_true')
    args = parser.parse_args()

    inputImgName = args.inputImgName
    outputLandmarksName = args.outputLandmarksName
    img = io.imread(inputImgName)
    shape = runDetector(img)
    writeShape(shape, outputLandmarksName)
    if args.show:
        show(img, shape)
    #print shape

