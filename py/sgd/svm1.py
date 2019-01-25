import numpy as np
from sgd import SGDMomentum, NAG, Adagrad
from scipy.optimize import minimize

clambda = 100.0

def svmGrad(weights, sample, target):
    if(target*weights.dot(sample) > 1.0):
        return clambda * weights
    else:
        return clambda*weights - target * sample

def svmLoss(weights, data, targets):
    res = 0.0
    res += clambda*weights.dot(weights)
    for i in xrange(data.shape[0]):
        res += max(0.0, 1.0 - targets[i]*weights.dot(data[i, :]))
    return res

if __name__ == '__main__':
    data = np.array([
        [1, 3, 1],
        [1.1, 5, 1],
        [1.1, 1.5, 1],
        [5, 7, 1],
        [4, 5, 1],
        [3, 3.2, 1],
        [2, 2.0, 1],
        [5, 3, 1],
        [3, 1, 1],
        [1.8, 1.5, 1]
        ])
    #data /= np.sum(data, axis=0)
    labels = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1])

    x = NAG(
        svmLoss, svmGrad, np.array([-5.86206898,  6.55172414, -2.37931031116], dtype=np.float32), 0.00005, 0.2, 240, data, labels)
    print x
    print svmLoss(x, data, labels)
    #print Adagrad(
        #svmLoss, svmGrad, np.array([0, 0, 0], dtype=np.float32), 0.002, 700, data, labels)

    res = minimize(lambda x: svmLoss(x, data, labels), np.array([0, 0, 0], dtype=np.float32))
    print res
    for i in xrange(data.shape[0]):
        #weights = res['x']
        weights = x
        print labels[i], weights.dot(data[i, :])

