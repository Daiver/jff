import numpy as np
import random

def SimpleSGD(funcForAll, gradForOne, weights, learningRate, nIters, data, targets):
    err = funcForAll(weights, data, targets)
    print 'err', err
    indices = range(data.shape[0])
    random.shuffle(indices)
    for iter in xrange(nIters):
        for sIter in xrange(data.shape[0]):
            index = indices[sIter]#random.randrange(0, data.shape[0])
            grad = gradForOne(weights, data[index, :], targets[index])
            weights -= grad*learningRate

        err = funcForAll(weights, data, targets)
        print iter, 'err', err
    return weights

def NAG(
        funcForAll, 
        gradForOne, 
        weights, 
        learningRate, 
        momentumWeight, 
        nIters, 
        data, targets):
    err = funcForAll(weights, data, targets)
    minErr = 1e10
    bestWeights = weights
    print 'err', err
    grad = np.zeros(weights.shape)
    indices = range(data.shape[0])
    for iter in xrange(nIters):
        for sIter in xrange(data.shape[0]):
            #index = random.randrange(0, data.shape[0])
            index = indices[sIter]#random.randrange(0, data.shape[0])
            grad = gradForOne(
                    weights - momentumWeight * learningRate * grad, 
                    data[index, :], targets[index]) + grad*momentumWeight
            weights -= grad*learningRate

        err = funcForAll(weights, data, targets)
        if err < minErr:
            bestWeights = np.copy(weights)
            minErr = err
            #print 'min', minErr, bestWeights
        #minErr = min(err, minErr)
        print iter, 'err', err, minErr
    return bestWeights


def SGDMomentum(
        funcForAll, 
        gradForOne, 
        weights, 
        learningRate, 
        momentumWeight, 
        nIters, 
        data, targets):
    err = funcForAll(weights, data, targets)
    minErr = err
    print 'err', err
    grad = np.zeros(weights.shape)
    indices = range(data.shape[0])
    for iter in xrange(nIters):
        for sIter in xrange(data.shape[0]):
            #index = random.randrange(0, data.shape[0])
            index = indices[sIter]#random.randrange(0, data.shape[0])
            grad = gradForOne(weights, data[index, :], targets[index]) + grad*momentumWeight
            weights -= grad*learningRate

        err = funcForAll(weights, data, targets)
        minErr = min(err, minErr)
        print iter, 'err', err, minErr
    return weights

def Adagrad(
        funcForAll, 
        gradForOne, 
        weights, 
        learningRate, 
        nIters, 
        data, targets):
    err = funcForAll(weights, data, targets)
    minErr = err
    print 'err', err
    grad = np.zeros(weights.shape)
    indices = range(data.shape[0])
    dumpFactor = 0
    for iter in xrange(nIters):
        for sIter in xrange(data.shape[0]):
            #index = random.randrange(0, data.shape[0])
            index = indices[sIter]#random.randrange(0, data.shape[0])
            grad = gradForOne(weights, data[index, :], targets[index]) 
            dumpFactor += sum(grad**2)
            weights -= grad*learningRate*1.0/np.sqrt(dumpFactor)

        err = funcForAll(weights, data, targets)
        minErr = min(err, minErr)
        print iter, 'err', err, minErr, 1.0/np.sqrt(dumpFactor)
    return weights

