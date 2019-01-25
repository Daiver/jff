import numpy as np
from scipy.optimize import rosen, rosen_der, minimize
import plot_rosen
from linesearch import lineSearch

def gainRatio(oldF, newF, oldL, newL):
    return (newF - oldF)/(newL - oldL)

def sr1Update(prevB, gradDiff, argDiff):
    predErr = gradDiff - prevB.dot(argDiff)
    if np.linalg.norm(predErr) < 0.00001:
        #print "Good enough B"
        return prevB
    denom = predErr.T.dot(argDiff)
    if np.linalg.norm(denom) < 0.00001:
        #print "bad denom"
        return prevB
    return prevB + (predErr.dot(predErr.T)/denom)

def rudeSR1(func, grad, initialX):
    xs, ys, zs = [], [], []
    initialX = np.array(initialX)
    B = np.identity(initialX.shape[0]) * 1
    x0 = initialX
    g0 = grad(x0)
    #print B
    nIter = 100
    for iter in xrange(nIter):
        direction = -np.linalg.pinv(B + np.identity(x0.shape[0]) * 1.0).dot(g0)
        stepLength = lineSearch(1.0, 0.5, 0.1, 100, func, g0, x0, direction)
        x1 = x0 + stepLength * direction

        g1 = grad(x1)
        B = sr1Update(B, g1 - g0, x1 - x0)
        g0 = g1
        x0 = x1
        err = func(x0)
        print 'Err', err, stepLength, direction
        #print B
        xs.append(x0[0])
        ys.append(x0[1])
        zs.append(err + 1)

    print B
    print 'res', x0
    return xs, ys, zs

if __name__ == '__main__':
    res = rudeSR1(rosen, rosen_der, [0.2,  2.0])
    #res = rudeSR1(rosen, rosen_der, [-1.2,  1.1])
    #print res
    plot_rosen.plotRosenbrock(res)

    #print minimize(rosen, [2.0, 1.0], 
                   ##method='Newton-CG', 
                   #method='bfgs', 
                   ##method='nelder-mead', 
                   #options={'disp': True})
