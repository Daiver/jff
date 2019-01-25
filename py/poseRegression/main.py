import cv2
import numpy as np
import random
import os
import pickle
import timeit
import sys
import cProfile
import copy

from common import applyTransform, draw_arrow

import dataprocessing
from dataprocessing import loadData, makeTrainDataFromRawData, makeTrainDataFromRawDataTriangles, computePoseFromTriangle, apply5DTransformation

import CPR

#def applyTransform(transformation, point, scale=1):
    #return (point + transformation)*scale

def coordToTuple(coord):
    return coord[0], coord[1]

class PosePart:
    def __init__(self, initialTransformation):
        #self.transformationMatrix = np.eye(3)
        self.transformation = initialTransformation
#        self.poseIndexedFeatures = [
                #((10, 5), (11, 2)),
                #((-15, -7), (-6, -7))
                #]
        self.regressor = None

    def draw(self, img):
        img = cv2.pyrUp(img)
        #img = cv2.pyrUp(img)
        defC = np.array([0, 0])
        defX = np.array([30, 0])
        defY = np.array([0, 30])

        showScale = 2

        #if len(img.shape) == 2:
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        colors = [
                    (0, 200, 0),
                    (200, 0, 0),
                    (0, 0, 200),
                    (0, 128, 255),
                    (200, 150, 110),
                    (170, 170, 170)
                ] * 10

        draw_arrow(img, 
                coordToTuple(applyTransform(self.transformation, defC, showScale)), 
                coordToTuple(applyTransform(self.transformation, defX, showScale)), 
                (10, 200, 200), 5)

        draw_arrow(img, 
                coordToTuple(applyTransform(self.transformation, defC, showScale)), 
                coordToTuple(applyTransform(self.transformation, defY, showScale)), 
                (10, 200, 200), 5)

#        for i, (p1, p2) in enumerate(self.poseIndexedFeatures):
            #draw_arrow(img, 
                #coordToTuple(applyTransform(self.transformation, defC, showScale)), 
                #coordToTuple(applyTransform(self.transformation, p1, showScale)), 
                #colors[i], 3)
            #draw_arrow(img, 
                #coordToTuple(applyTransform(self.transformation, defC, showScale)), 
                #coordToTuple(applyTransform(self.transformation, p2, showScale)), 
                #colors[i], 3)
        return img

if __name__ == '__main__':

    np.set_printoptions(precision=4)

    images, bboxes, values1, values2, values3, names = loadData('/home/daiver/Downloads/COFW/dump/')
    print 'All data count', len(images)
    random.seed(42)
    if len(sys.argv) == 1:

        testSamplesCount = 60
        imagesTr  = images[:-testSamplesCount]
        bboxesTr  = bboxes[:-testSamplesCount]
        values1Tr = values1[:-testSamplesCount]
        values2Tr = values2[:-testSamplesCount]
        values3Tr = values3[:-testSamplesCount]

        print 'Train images count', len(imagesTr)

        #trainData, trainValues = makeTrainDataFromRawData(imagesTr, bboxesTr, valuesTr, 2, 10)
        trainData, trainValues = makeTrainDataFromRawDataTriangles(
                imagesTr, bboxesTr, 
                values1Tr, values2Tr, values3Tr, 100)
        print 'Train data length', len(trainData)
        start_time = timeit.default_timer()
        #cascade, points = cProfile.run('CPR.fitCascade4(trainData, trainValues, 64, 5, 40, 2)')
        cascade, points = CPR.fitCascade4(
                copy.deepcopy(trainData), trainValues, 464, 5, 964, 10)
        #cascade, points = CPR.fitCascade3(trainData, trainValues, 7, 64, 50)
        elapsed = timeit.default_timer() - start_time
        print 'Elapsed', elapsed
        f = open('cascade.dump', 'w')
        pickle.dump((cascade, points), f)
        f.close()
    else:
        cascade, points = pickle.load(open(sys.argv[1]))

    sumOfErrors = 0.0
    countOfErrors = 0

    #for testIndex in xrange(len(images)):
    for testIndex in xrange(len(trainData)):
        #testIndex = 2
        #img = images[testIndex]
        img = trainData[testIndex][0]
        dataIndex = trainData[testIndex][2]

        p1 = values1[dataIndex]
        p2 = values2[dataIndex]
        p3 = values3[dataIndex]
        pose = computePoseFromTriangle(p1, p2, p3)
        bbox = bboxes[dataIndex]
        sample = dataprocessing.randomSampleFromBBox(bbox)
        #sample = trainData[testIndex][1]

        print "Index", testIndex
        print 'initialTransformation', sample
        res = CPR.activateCascade4(cascade, points, img, sample)
        #res = CPR.poseClustering(cascade, points, images[testIndex], bboxes[testIndex], 16)
        ans = pose
        err = np.linalg.norm(res - ans)
        if err > 5:
            countOfErrors += 1
        sumOfErrors += err
        print 'Res', res
        print 'True ans', ans

        dp1 = apply5DTransformation(pose, dataprocessing.defaultP1)
        dp2 = apply5DTransformation(pose, dataprocessing.defaultP2)
        dp3 = apply5DTransformation(pose, dataprocessing.defaultP3)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(img, (int(dp1[0]), int(dp1[1])), 5, (0, 255, 0))
        cv2.circle(img, (int(dp2[0]), int(dp2[1])), 5, (0, 255, 0))
        cv2.circle(img, (int(dp3[0]), int(dp3[1])), 5, (0, 255, 0))

        rp1 = apply5DTransformation(res, dataprocessing.defaultP1)
        rp2 = apply5DTransformation(res, dataprocessing.defaultP2)
        rp3 = apply5DTransformation(res, dataprocessing.defaultP3)

        ip1 = apply5DTransformation(sample, dataprocessing.defaultP1)
        ip2 = apply5DTransformation(sample, dataprocessing.defaultP2)
        ip3 = apply5DTransformation(sample, dataprocessing.defaultP3)

        cv2.circle(img, (int(rp1[0]), int(rp1[1])), 5, (255, 0, 0))
        cv2.circle(img, (int(rp2[0]), int(rp2[1])), 5, (255, 0, 0))
        cv2.circle(img, (int(rp3[0]), int(rp3[1])), 5, (255, 0, 0))

        cv2.circle(img, (int(ip1[0]), int(ip1[1])), 5, (0, 0, 255))
        cv2.circle(img, (int(ip2[0]), int(ip2[1])), 5, (0, 0, 255))
        cv2.circle(img, (int(ip3[0]), int(ip3[1])), 5, (0, 0, 255))

        #cv2.circle(img, (int(res[0]), int(res[1])), 5, (255, 0, 0))
        #cv2.circle(img, (int(initialTransformation[0]), int(initialTransformation[1])), 5, (0, 0, 255))
        cv2.imshow('', img)
        cv2.imwrite('dump/' + str(testIndex) + '.png', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    print 'sumOfErrors', sumOfErrors, countOfErrors
