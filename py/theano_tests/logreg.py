import numpy as np
import theano
import theano.tensor as T
import input_data
from sklearn.preprocessing import normalize
from scipy.optimize import minimize

from sgd import SGDMomentum

print "Imported"

rng = np.random

def sigmoid(z):
    return 1.0/(1.0 + T.exp(-z))

def layerAct(actFunc, weights, sample):
    linAct = T.dot(weights[:,:-1], sample) + weights[:, -1]
    return actFunc(linAct)

def squareLoss(function, data, targets):
    scanRes, _ = theano.scan(function, sequences=data)
    return T.sum((scanRes - targets)**2)

def getCostFunction():
    data    = T.matrix("data")
    targets = T.vector("targets")
    weights = T.matrix("weights")
    sample  = T.vector("sample")

    actFunc = lambda sample: layerAct(sigmoid, weights, sample)[0]
    costFunc = squareLoss(actFunc, data, targets)

    #theano.printing.debugprint(costFunc)
    cost = theano.function([weights, data, targets], costFunc)
    grad = theano.function([weights, data, targets], T.grad(costFunc, [weights]))
    func = theano.function([weights, sample], actFunc(sample))
    return func, cost, grad

def main1():
    func, costFunc, gradFunc = getCostFunction()
    print 'Constructed!'

    data = np.array([
        [0.1, 0.2],
        [0.1, 0.2],
        [3.1, 1.2],
        ], dtype=theano.config.floatX)

    targets = np.array([0, 0, 1.0])

    weights = np.array([[0.1, 0.0, 0.0]], dtype=theano.config.floatX)

    for iter in xrange(1000):
        err = costFunc(weights, data, targets)
        grad = gradFunc(weights, data, targets)[0]
        #print 'iter', iter, err
        #print grad
        weights -= 0.05 * grad

    print costFunc(weights, data, targets)

    print weights
    for i in xrange(data.shape[0]):
        print func(weights, data[i, :]), targets[i]

def main2():
    func, costFunc, gradFunc = getCostFunction()
    print 'Constructed!'

    mnist = input_data.read_data_sets("/home/daiver/Downloads/MNIST_data/", one_hot=True)
    data    = mnist.test.images
    data    = normalize(data)
    targets = mnist.test.labels
    print 'targets shape', targets.shape
    targets = targets[:, 5]

    weights = np.random.random(data.shape[1] + 1).reshape((1, -1))#, dtype=np.float32)

    weights = SGDMomentum(costFunc, gradFunc, weights,
            0.1, 0.9, 1000, data, targets)

#    for iter in xrange(1000):
        #err = costFunc(weights, data, targets)
        #grad = gradFunc(weights, data, targets)[0]
        #print 'iter', iter, err
        ##print grad
        #weights -= 0.05 * grad

    print costFunc(weights, data, targets)


    nErr = 0
    for i in xrange(data.shape[0]):
        res = func(weights, data[i, :])
        ans = targets[i]
        if i < 10: print i, res, ans
        if (res >= 0.5 and ans == 0) or (res <= 0.5 and ans == 1):
            nErr += 1
    print 'nErr', nErr

if __name__ == '__main__':
    main2()

