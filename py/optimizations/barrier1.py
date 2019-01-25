import numpy as np
from scipy.optimize import minimize

def barrierMethod(func, constraints, initT, mu, nIters, vars, verbose):
    t = initT
    for iter in xrange(nIters):
        f = lambda x: t*func(x) - sum(np.log(max(0.000000001, -c(x))) for c in constraints)
        res = minimize(f, vars, method='SLSQP')
        if verbose:
            print 'Err', f(vars)
            print res
            print 't', t
        vars = res['x']
        t = t*mu
        if 1.0/t < 0.0001:
            break
    return vars

def svmTest():
    data = np.array([
        [1, 3],
        [1.1, 5],
        [1.1, 1.5],
        [5, 7],
        [4, 5],
        [3, 3.2],
        [2, 2.0],
        [5, 3],
        [3, 1],
        [1.8, 1.5]
        ])
    labels = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
    func = lambda x: x[:-1].dot(x[:-1])
    constraints = []
    for x, l in zip(data, labels):
        l1 = l
        x1 = np.copy(x)
        constraints.append(lambda a: -l1*(a[:-1].dot(x1) + x1[-1]) + 1)
    print barrierMethod(
            func, constraints,
            0.00001, 1.1, 1000, 
            np.array([0.1, 0.1, 0.1]), True)

if __name__ == '__main__':
    svmTest()
    #func = lambda x: x[0]*x[0]
    #constraints = [lambda x: x[0] + 1]
    #print barrierMethod(func, constraints, 0.01, 1.2, 10, [-1.2], True)
    #print np.log(-constraints[0]([-1.2]))
