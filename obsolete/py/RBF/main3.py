import numpy as np
import scipy.sparse.linalg

from matplotlib import pyplot as plt

def gaussian(s2, r2):
    return np.exp(-s2/r2)

rbfFunc = gaussian

def computeRBFCoeffs(X, Y, radius, damp=0.0):
    Y = np.array(Y, dtype=np.float32)
    A = np.zeros((len(X), len(X)))
    for i, x1 in enumerate(X):
        for j, x2 in enumerate(X):
            A[i, j] = rbfFunc((x1 - x2)**2, radius**2)
    alpha = np.linalg.lstsq(A, Y)[0]
    #alpha = scipy.sparse.linalg.lsmr(A, Y, damp)[0]
    #alpha = scipy.sparse.linalg.lsqr(A, Y, damp)[0]
    return alpha

def RBF1D(X, coeffs, radius):
    return lambda x1:(
        sum(a*rbfFunc((x - x1)**2, radius**2) for a, x in zip(coeffs, X)) )

def iterativeRBF1DCoeffs(X, initY, radiuses, damps):
    Y = np.array(initY, dtype=np.float32)
    print Y
    res = []
    for k, (r, d) in enumerate(zip(radiuses, damps)):
        alpha = computeRBFCoeffs(X, Y, r, d)
        rbfTmp = RBF1D(X, alpha, r)
        for i, x in enumerate(X):
            Y[i] -= rbfTmp(x)
        print k, r, d
        print Y
        res.append(alpha)

    return res

def iterativeRBF1D(X, radiuses, coeffs):
    rbfs = [RBF1D(X, c, r) for c, r in zip(coeffs, radiuses)]
    return lambda x:(sum(rbf(x) for rbf in rbfs))

if __name__ == '__main__':
    points  = [0, 1,   2, 3, 4]
    targets = [0, 1,  -1, 0, 0]

    #points  = [-1, 0, 1,  2, 3, 4]
    #targets = [ 0, 0, 2, -2, 0, 0]

    radius1 = 3.0
    
    #radiuses = [3.5,   3.0,   2.5,   2.0,   1.5,   1.0,  0.7,  0.5, 0.2 ][:]
    #damps    = [0.005, 0.005, 0.05, 0.05, 0.005, 0.005, 0.05, 0.05, 0.05][:]

    alpha = computeRBFCoeffs(points, targets, radius1, 0.00)
    #alphas = iterativeRBF1DCoeffs(points, targets, 
    #        radiuses, damps)

    print alpha
    rbf1 = RBF1D(points, alpha, radius1)
    #rbf2 = iterativeRBF1D(points, radiuses, alphas)
    print 'Test'
    for i, x in enumerate(points):
        print x, rbf1(x), targets[i]
    print 'linspace'
    x = np.linspace(-1, 5, num=200)
    y1 = map(rbf1, x)
    #y2 = map(rbf2, x)
    #print min(y2), max(y2)
    plt.plot(x, y1, '--', points, targets, 'ro')
    plt.show()
    #for x in np.linspace(-1, 4, num=50):
    #    print x, rbf(x)

