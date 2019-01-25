import numpy as np

def softthresholding(b, clambda):
    if   b > clambda:
        return b - clambda
    elif b < -clambda:
        return b + clambda
    return 0

def softthresholdingVec(bVec, clambda):
    res = np.copy(bVec)
    for i in xrange(len(bVec)):
        res[i] = softthresholding(bVec[i], clambda)
    return res
