import numpy as np
import random

def SGDMomentum(
        funcForAll, 
        gradForOne, 
        weights, 
        learningRate, 
        momentumWeight, 
        nIters, 
        data, targets):
    #err = funcForAll(weights, data, targets)
    #minErr = err
    #print 'err', err
    grad = np.zeros(weights.shape)
    indices = range(data.shape[0])
    smpl = np.zeros((1, data.shape[1]))
    trgt = np.zeros((1))
    for iter in xrange(nIters):
        for sIter in xrange(data.shape[0]):
            index = indices[sIter]#random.randrange(0, data.shape[0])
            smpl[0] = data[index, :]
            trgt[0] = targets[index]
            tmp = gradForOne(weights, smpl, trgt)[0]
            grad = tmp + grad*momentumWeight
            #del tmp
            #del smpl 
            #del trgt
            weights -= grad*learningRate
            print iter, sIter, funcForAll(weights, smpl, trgt)

        #err = funcForAll(weights, data, targets)
        #minErr = min(err, minErr)
        #print iter, 'err', err, minErr
    return weights

