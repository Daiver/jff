import numpy as np

def findBestThr(values, weights, labels):
    #assert False
    nNegLeft  = 0
    nPosLeft  = 0
    nNegRight = 0
    nPosRight = 0

    #indices = range(0, len(labels))

    indices = np.argsort(values)

    for i in indices:
        l = labels[i]
        w = weights[i]
        v = values[i]
        if l == 1:
            nPosRight += w
        else:
            nNegRight += w

    bestErr = nNegRight
    bestPolarity = 1.0
    if nPosRight < nNegRight:
        bestErr = nPosRight
        bestPolarity = -1.0
    err     = bestErr
    bestThr = 0

    #print indices
    for i1, i in enumerate(indices):
        #print i, nNegLeft, nPosLeft, nNegRight, nPosRight, err
        l = labels[i]
        w = weights[i]
        v = values[i]
        if l == 1:
            nPosRight -= w
            nPosLeft  += w
        else:
            nNegRight -= w
            nNegLeft  += w
        err1 = nPosLeft + nNegRight
        err2 = nPosRight + nNegLeft
        err = min(err1, err2)
        if err < bestErr:
            bestErr = err
            if i1 < len(indices) - 1:
                bestThr = 0.5 * (values[i] + values[indices[i1 + 1]])
            else:
                bestThr = v

            if err1 < err2:
                bestPolarity = 1.0
            else:
                bestPolarity = -1.0

    return bestThr, bestErr, bestPolarity

