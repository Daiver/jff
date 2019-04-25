import numpy as np
from scipy.optimize import rosen, rosen_der, minimize
import plot_rosen

def lineSearch(initialA, rho, c, maxIter, func, gradVal, x0, direction):
    f0 = func(x0)
    a = initialA
    funcVals = [f0]
    alphas   = [0.0]
    for iter in xrange(maxIter):
        funcVal = func(x0 + a * direction)
        funcVals.append(funcVal)
        alphas.append(a)
        if funcVal <= (f0 + c * a * (gradVal.dot(direction))):
            return a
        a = rho * a
    return a

def quadLineSearch(delta, x0, grad0, direction, grad):
    nom   = grad0.dot(direction)
    denom = grad(x0 + delta * direction).dot(direction) - nom
    return - delta * nom/denom

def quadLineSearchIter(nIter, toleranceSquared, delta, x0, direction, grad):
    x = x0
    directionSq = direction.dot(direction)
    alpha       = -delta
    nuPrev      = grad(x + delta * direction).dot(direction)
    for iter in xrange(nIter):
        nu     = grad(x).dot(direction)
        alpha  = alpha * nu/(nuPrev - nu)
        x      = x + alpha * direction
        nuPrev = nu
        if alpha**2 * directionSq < toleranceSquared:
            break
    return alpha

def gradDescent(func, grad, initialX, maxIter):
    x = initialX
    for iter in xrange(maxIter):
        funcVal = func(x)
        direction  = -grad(x)
        #stepLength = quadLineSearchIter(5, 0.1, 0.000001,
                                    #x, direction, grad)
        stepLength = quadLineSearch(0.00001, x, -direction, direction, grad)
        #stepLength = lineSearch(
                #1.0, 0.5, 0.0001, 30, func, -direction, x, direction)
        x += stepLength * direction
        err = func(x)
        print iter, 'err', err, x, stepLength, direction
        if abs(err) < 0.000001:
            break

    return x

if __name__ == '__main__':
    #print gradDescent(
            #lambda x: x[0]**2 + x[1]**2,
            #lambda x: np.array([x[0], x[1]]),
            #np.array([10, 20]),
            #10)

    print gradDescent(
            rosen,
            rosen_der,
            np.array([-0.1, -1.0]),
            100)
