from softthresholding import softthresholdingVec
from proxgrad import proximalGradient, acceleratedProximalGradient

from lassocd import lassoCoordinateDescent
import numpy as np
import cvxpy

import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.5f s' % (f.func_name, (time2-time1))
        return ret
    return wrap

def runndCVXPy(A, b, clambda):
    n_vars = A.shape[1]
    x = cvxpy.Variable(n_vars)
    obj = cvxpy.Minimize(
            clambda * cvxpy.sum_entries(cvxpy.abs(x)) +
            0.5     * cvxpy.sum_entries(cvxpy.square(A * x - b)))
    prob = cvxpy.Problem(obj, [])
    prob.solve(verbose=False)
    return x.value

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

def lassoProxGrad(A, b, clambda, verbose = False):

    def mkL1Prox(clambda):
        def inner(v, step):
            #return v
            return softthresholdingVec(v, step * clambda)
        return inner

    def mkFunc(A, b):
        def inner(x):
            return 1.0/2.0 * np.sum(np.square(A.dot(x) - b))
        return inner

    def mkGrad(A, b):
        AtA = A.T.dot(A)
        Atb = A.T.dot(b)
        def inner(x):
            return AtA.dot(x) - Atb
        return inner

    func = mkFunc(A, b)
    grad = mkGrad(A, b)
    prox = mkL1Prox(clambda)
    x = np.zeros(A.shape[1])
    #initialStepSize = 1.5
    initialStepSize = 0.0456530823607
    #initialStepSize = 0.04
    #return acceleratedProximalGradient(func, grad, prox, 0.5, initialStepSize, x, verbose)
    #return proximalGradient(func, grad, prox, 1.0, initialStepSize, x, verbose)
    return proximalGradient(func, grad, prox, 0.5, initialStepSize, x, verbose)

@timing
def test(customLasso, cvxCheck = True):
    for iter in xrange(1000):
        n_residuals = 15000
        n_vars = 20
        A = np.random.uniform(-5, 5, size=[n_residuals, n_vars])
        b = np.random.uniform(-5, 5, size=[n_residuals])
        clambda = np.random.uniform(0, 10)
        res = customLasso(A, b, clambda)
        if not cvxCheck:
            continue
        ans = runndCVXPy(A, b, clambda).reshape(-1)
        if np.sum(np.abs(res - ans)) > 1e-3:
            print 'wrong?'
            print 'A'
            print A
            print 'b'
            print b, 
            print 'clambda', clambda, 
            print 'res', res
            print 'ans', ans

if __name__ == '__main__':
    A = np.array([
            [0, 1],
            [2, 1],
            [4, 1]
        ])
    AtA = A.T.dot(A)
    eigA = max(np.linalg.eigvalsh(AtA))
    print 1.0/eigA
    b = [1, 2, 3]
    clambda = 0.5
    print runndCVXPy(A, b, clambda).reshape(-1)
    print lassoCoordinateDescent(A, b, clambda, True)
    print lassoProxGrad(A, b, clambda, True)
    exit(0)
    checkSolution = False
    #checkSolution = True
    print 'Test Coord descent'
    test(lassoCoordinateDescent, checkSolution)
    print 'Test02'
    test(lassoProxGrad, checkSolution)
    
