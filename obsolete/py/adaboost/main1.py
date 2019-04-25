import numpy as np
import matplotlib.pyplot as plt
import adaboost

import dec_stump
import w_dec_stump

def eval(clf, data, labels):
    err = 0.0
    for i, x in enumerate(data):
        ans = clf.activate(x)
        if ans != labels[i]:
            err += 1 
    return err/data.shape[0]

def genData(nSamples):
    minV, maxV = 0, 10
    res = np.zeros((nSamples, 2))
    for i in xrange(nSamples):
        res[i] = [
                minV + np.random.ranf() * (maxV - minV),
                minV + np.random.ranf() * (maxV - minV)
                ]
    return res

def genLabels(data, center, rad):
    res = np.zeros(data.shape[0])
    for i in xrange(res.shape[0]):
        if ((data[i, 0] - center[0])**2 + (data[i, 1] - center[1])**2) < rad**2:
            res[i] = 1
        else:
            res[i] = 0

    return res

if __name__ == '__main__':
    data = genData(1000)
    radius = 2.9
    center = [5.0, 5.0]
    labels = genLabels(data, center, radius)
    stump = dec_stump.makeStump(data, labels, 2)

#    weights = np.ones(data.shape[0])/data.shape[0]
    ##weights[0] = 0.6
    ##weights = weights/sum(weights)
    ##print 'weights', weights
    #wStump = w_dec_stump.makeStumpWeighted(data, weights, labels, 2)

    #print 'Err', eval(stump, data, labels)
    #print stump.attr
    #print stump.val
    #print stump.freqsL
    #print stump.freqsR
    #print 'weighted'
    #print wStump.attr
    #print wStump.val
    #print wStump.freqsL
    #print wStump.freqsR
    #exit(0)

    abClf = adaboost.buildAdaBoost(
            lambda data, weights, labels: w_dec_stump.makeStumpWeighted(data, weights, labels, 2), 
            data, labels, 200)
    print 'ada'
    print abClf.clfWeights

    d0 = np.array([data[i] for i in xrange(data.shape[0]) if labels[i] == 0])
    d1 = np.array([data[i] for i in xrange(data.shape[0]) if labels[i] == 1])
    r0 = []
    r1 = []
    r2 = []
    for i in np.linspace(0, 10, 200):
        for j in np.linspace(0, 10, 200):
            if (i - center[0])**2 + (j - center[1])**2 < radius**2:
                r2.append([i, j])
            ans = abClf.activate([i, j])
            if ans == 0:
                r0.append([i, j])
            else:
                r1.append([i, j])

    r0 = np.array(r0)
    r1 = np.array(r1)
    r2 = np.array(r2)
    print len(r0), len(r1)
    #print r1
    #print d0
    #print 
    #print d1
    plt.plot(
            r0[:, 0], r0[:, 1], 'r.',
            r1[:, 0], r1[:, 1], 'g.',
            d0[:, 0], d0[:, 1], 'rD',
            d1[:, 0], d1[:, 1], 'gD',
            )
    #plt.plot(d1[:, 0], d1[:, 1], 'g^')
    plt.show()
