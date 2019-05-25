import numpy as np
import matplotlib.pyplot as plt
import random

def ransac(
        trainModel,
        testModel,
        data,
        nSamplesToTrain,
        nIters,
        goodFitTreshold,
        nMinItemsForGoodFit,
        verbose = True):
    bestModel = None
    bestError = 1e50
    for iter in xrange(nIters):
        hypInliers = random.sample(data, nSamplesToTrain) 
        model = trainModel(hypInliers)
        errors = map(lambda x: testModel(model, x), data)
        inliers = [x for i, x in enumerate(data) if testModel(model, x) < goodFitTreshold]
        if verbose:
            print 'iter', iter, 'nInliers', len(inliers), 'bestErr', bestError
        if len(inliers) >= nMinItemsForGoodFit:
            model = trainModel(inliers)
            err   = sum(map(lambda x: testModel(model, x), inliers))/len(inliers)
            if verbose:
                print err
            if err < bestError:
                bestModel = model
                bestError = err
                if bestError < 0.0000001:
                    break
    return bestModel, bestError

def activateLinReg(weights, x):
    return weights[:-1].dot(x) + weights[-1]

def trainLinRegModel(data):
    data = np.array(data)
    trainData = np.hstack((data[:, :-1], np.ones((len(data), 1))))
    trainValues = data[:, -1]
    res = np.linalg.lstsq(trainData, trainValues)[0]
    return res

def testLinRegModel(weights, sample):
    residual = activateLinReg(weights, sample[:-1]) - sample[-1]
    return residual**2

if __name__ == '__main__':
    points = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],

            [3, 1],
            [4, 2],
            [6, 4],
            [3, 6],
            [5, 7],
            [7, 8],
            [6, 7.5],
            [3, 6],
            [1, 5],
            [0.5, 4.75],
            [0, 4.5],

            [3, 6],
            [2, 8],
            [2.5, 7],
            [5, 2],
            [6, 0],
            [5.5, 1],
            [3.5, 7],
            [4.5, 3]
            ], dtype=np.float32)
    points += np.random.random(points.shape) * 0.09
    model, err = ransac(trainLinRegModel, testLinRegModel, points, 4, 500, 2.7, 7, True)
    #model = np.array([0.5, 4.5])
    predictedPoints = np.array([
            [points[i][0], activateLinReg(model, points[i][:-1])]
            for i in xrange(len(points))
            ])

    plt.plot(points[:, 0], points[:,1], 'or', predictedPoints[:, 0], predictedPoints[:, 1], '-g', markersize=10.0)
    plt.show()
