import numpy as np
from scipy.optimize import minimize
from sgd import SimpleSGD, SGDMomentum, Adagrad, NAG

def lseGrad(weights, sample, targetVal):
    return 2.0*(weights.dot(sample) - targetVal)*sample

def lseErr(weights, data, targets):
    res = 0
    for i in xrange(data.shape[0]):
        res += (weights.dot(data[i, :]) - targets[i])**2
    return res

if __name__ == '__main__':
    data = np.array([
        [0.9, 1],
        [2.1, 1],
        [3.2, 1],
        [4, 1]
        ], dtype=np.float32)
    targets = np.array([1.8, 3, 4, 5], dtype=np.float32)
    #print SimpleSGD(lseErr, lseGrad, np.array([0, 0], dtype=np.float32), 0.005, 500, data, targets)
    #print SGDMomentum(
            #lseErr, lseGrad, np.array([0, 0], dtype=np.float32), 0.005, 0.9, 80, data, targets)
    print NAG(
            lseErr, lseGrad, np.array([0, 0], dtype=np.float32), 0.005, 0.9, 80, data, targets)
    #print Adagrad(
            #lseErr, lseGrad, np.array([0, 0], dtype=np.float32), 1, 200, data, targets)
    print minimize(lambda x: lseErr(x, data, targets), np.array([0, 0], dtype=np.float32))
