import dlib
from skimage import io
import numpy
import sys
import os
import cv2
import numpy as np

predictorPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_68_face_landmarks.dat")

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorPath)

def runDetector(img):
    dets = detector(img, 1)
    if len(dets) != 1:
        print 'WARNING!', len(dets)
    shapes = []
    if len(dets) == 0:
        return shapes
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        shape = [[i.x, i.y] for i in shape.parts()]
        shape = numpy.array(shape)
        shapes.append(shape)
    return shapes

def show(img, shape):
    import matplotlib.pyplot as plt
    #import cv2
    #cv2.imshow('', img)
    #cv2.waitKey()
    plt.imshow(img)
    plt.scatter(shape[:, 0], shape[:, 1], c='w', s=8)
    plt.show()

def drawAndSaveCv2(img, shapes, outImgName):
    img = np.array(img[:, :, ::-1]) #cv2format
    for shape in shapes:
        for p in shape:
            p = map(int, p)
            cv2.circle(img, (p[0], p[1]), 3, (0, 255, 0))
    img = cv2.pyrUp(cv2.pyrUp(img))
    cv2.imwrite(outImgName, img)

def writeShape(shape, outputLandmarksName):
    with open(outputLandmarksName, 'w') as f:
        for x in shape:
            f.write('%s %s\n' % (x[0], x[1]))

if __name__ == '__main__':
    DATA_ROOT = "/home/daiver/Downloads/FaceWarehouse/"
    nPersons = 150

    
    for personInd in xrange(1, nPersons + 1):
        personName = "Tester_%s" % (personInd)
        personDir  = os.path.join(DATA_ROOT, personName)
        subdirs = os.listdir(personDir)
        for subdir in subdirs:
            absSubdir = os.path.join(personDir, subdir)
            print absSubdir
            for poseInd in xrange(0, 24):
                imgName = os.path.join(absSubdir, "pose_%s.jpg"  % poseInd)
                lndName = os.path.join(absSubdir, "pose_%s.land" % poseInd)

                img = io.imread(imgName)
                shapes = runDetector(img) 
                outImgName = 'dump/' + str(personName) + "_" + subdir + "_" + str(poseInd) + ".png"
                drawAndSaveCv2(img, shapes, outImgName)
                #show(img, shape)
