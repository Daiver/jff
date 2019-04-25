import numpy as np
import linesearch

def nonLinCG(func, grad, initialX, nIter):
    x = initialX
    r0 = -grad(x)
    d0 = r0
    gradLen = r0.shape[0]
    curSteps = 0
    for iter in xrange(nIter):
        curSteps += 1
        #alpha = linesearch.quadLineSearchIter(
                #5, 0.00001, 0.0001, x, d0, grad)
        alpha = linesearch.quadLineSearch(0.000001, x, -r0, d0, grad)
        x     = x + alpha * d0
        r1    = -grad(x)
        beta  = r1.dot(r1 - r0)/(r0.dot(r0))
        if beta < 0:
            beta = 0
        elif curSteps > gradLen:
            curSteps = 0
            beta = 0
        d0 = r1 + beta * d0
        r0 = r1
        err = func(x)
        print iter, 'err', err, x
        if err < 0.00001:
            break
    return x

def nonLinCGSeq(func, grad, initialX, nIter):
    x = initialX
    xs = [x]
    errs = [func(x)]

    r0 = -grad(x)
    d0 = r0
    gradLen = r0.shape[0]
    curSteps = 0
    for iter in xrange(nIter):
        curSteps += 1
        #alpha = linesearch.quadLineSearchIter(
                #5, 0.00001, 0.0001, x, d0, grad)
        alpha = linesearch.quadLineSearch(0.0001, x, -r0, d0, grad)
        x     = x + alpha * d0
        r1    = -grad(x)
        beta  = r1.dot(r1 - r0)/(r0.dot(r0))
        if beta < 0:
            beta = 0
        elif curSteps > gradLen:
            curSteps = 0
            beta = 0
        d0 = r1 + beta * d0
        r0 = r1
        err = func(x)
        xs.append(x)
        errs.append(err)
        print iter, 'err', err, x
        if err < 0.00001:
            break
    return np.array(xs), np.array(errs)


if __name__ == '__main__':
    from scipy.optimize import rosen, rosen_der, minimize
    import plot_rosen
 
#    print nonLinCG(
            #rosen,
            #rosen_der,
            #np.array([-0.1, -1.0]),
            #50)
   
    xss, zs = nonLinCGSeq(
            rosen,
            rosen_der,
            np.array([-2.1, 1.0]),
            50)
    xs, ys = xss[:, 0], xss[:, 1]
    plot_rosen.plotRosenbrock([xs, ys, zs])

