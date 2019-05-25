import numpy as np
from softthresholding import softthresholding

def solveCoordDescent4OneVar(A, b, x, j, clambda):
    n_residuals, n_vars = A.shape
    rhs = np.zeros(n_residuals)
    for k in xrange(n_vars):
        if k == j:
            continue
        rhs += A[:, k] * x[k]
    rhs -= b
    rhs *= -1
    aVec = A[:, j]
    b2Thresholding = aVec.dot(rhs)
    a2Divide = aVec.dot(aVec)
    xUnnorm = softthresholding(b2Thresholding, clambda)
    #print '>', rhs
    return xUnnorm / a2Divide

def lassoCoordinateDescent(A, b, clambda, verbose=False):
    n_vars  = A.shape[1]
    x       = np.zeros(n_vars)
    n_iters = 30
    for iter in xrange(n_iters):
        x0 = np.copy(x)
        for varInd in xrange(n_vars):
            x[varInd] = solveCoordDescent4OneVar(A, b, x, varInd, clambda)
        dx = np.linalg.norm(x0 - x)**2
        if dx < 1e-6:
            break
        if verbose:
            print iter, 0.5*np.sum(np.square(A.dot(x) - b)) + clambda*np.sum(np.abs(x))
    return x


