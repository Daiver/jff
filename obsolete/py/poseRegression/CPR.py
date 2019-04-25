import numpy as np
import random
import copy
import sklearn.ensemble

import common
from common import applyTransform

from fern import FernRegressor, FernRegressorBoosted

import dataprocessing

def samplePointsPairsInSquare(nPoints, sideSize):
    res = []
    for i in xrange(nPoints):
        res.append(
             (np.array([random.uniform(-sideSize, sideSize), 
                        random.uniform(-sideSize, sideSize)]),
              np.array([random.uniform(-sideSize, sideSize), 
                        random.uniform(-sideSize, sideSize)])))
    return res

def transformPairsOfPoints(transformation, points):
    return [(dataprocessing.apply5DTransformation(transformation, p1),
             dataprocessing.apply5DTransformation(transformation, p2))
            for p1, p2 in points]

def diffFromTransform(img, transformation, points):
    return common.computeDifferences(img, transformPairsOfPoints(transformation, points))

'''
    TrainExample:
        img
        transformation
'''

def activateCascade4(cascade, cascadePoints, img, initialTransformation):
    trans = copy.deepcopy(initialTransformation)
    for stage, points in zip(cascade, cascadePoints):
        for i in xrange(len(initialTransformation)):
            ans = stage[i].predict(diffFromTransform(img, trans, points))[0]
            trans[i] += ans
        trans[2] = normalizeAngle(trans[2])
    return trans

def poseClustering(cascade, cascadePoints, img, bbox, countOfTryes):
    res = []
    for i in xrange(countOfTryes):
        sample = dataprocessing.randomSampleFromBBox(bbox)
        res.append(activateCascade4(cascade, cascadePoints, img, sample))

    return np.average(res, axis=0)
    
def normalizeAngle(angle):
    #return angle
    while angle >  np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

def fitCascade4(
        trainData,
        initialValues,
        countOfFeatures,
        fernDepth,
        fernsPoolSize,
        cascadeSize):
    cascade = []
    cascadePoints = []
    np.set_printoptions(precision=5)

    #print '\n\t'.join(str(t[1]) for t in trainData)
    for outerIter in xrange(cascadeSize):
        values = copy.deepcopy(initialValues)

        #print 'Prepare data'
        for i in xrange(len(values)):
            for j in xrange(len(values[0])):
                values[i][j] -= trainData[j][1][i]

        #print 'Sample points'
        points = samplePointsPairsInSquare(countOfFeatures, 1.0)

        #print 'Computing diffs'
        diffs = []
        for img, trans, _ in trainData:
            diff = diffFromTransform(img, trans, points)
            diffs.append(diff)

        currentStage = []
        err = 0
        for i in xrange(len(values)):
            #print "Fitting....", outerIter, i
            currentStage.append(FernRegressorBoosted(fernDepth, fernsPoolSize, 10))
            #currentStage.append(sklearn.ensemble.RandomForestRegressor())
            currentStage[-1].fit(diffs, values[i])
            err += sum(abs(currentStage[-1].predict(np.array(diffs)) - np.array(values[i])))

        cascade.append(currentStage)
        cascadePoints.append(points)
        #print transformPairsOfPoints(trainData[0][1], points)
        #print diffs
        #print 'after\n', '\t\n'.join(str(t[1]) for t in trainData)
        #print 
        #print '\n\t'.join(map(str, values))
        print outerIter, ': Err', err
        #print ''
        #print '\n'.join(map(str,currentStage))
        #print 'avg', map(np.average, values)
        #print ' '.join(str(t[1]) for t in trainData)
        for i in xrange(len(values)):
            for j, (img, trans, _) in enumerate(trainData):
                #values[i] -= bestFern.activate(diffFromTransform(img, trans, fern.points))
                diffs = diffFromTransform(img, trans, points)
                #print 'D', diffs
                trainData[j][1][i] += currentStage[i].predict(diffs)[0]
        for t in trainData:
            t[1][2] = normalizeAngle(t[1][2])
                #trainData[j][1][i] = normalizeAngle(trainData[j][1][i])
        #print 'after\n', '\t\n'.join(str(t[1]) for t in trainData)
        #print '\n\t'.join(map(str, values))

    return cascade, cascadePoints

