import numpy as np
from scipy.optimize import minimize, check_grad
import autograd

def tridFunc(x):
    d = x.shape[0]
    res = 0.0
    for i in xrange(d):
        res += (x[i] - 1.0)**2
    for i in xrange(1, d):
        res -= x[i]*x[i-1]
    return res

if __name__ == '__main__':
    d = 10
    x = np.random.rand(d)
    grad = autograd.grad(tridFunc)
    print check_grad(tridFunc, grad, x)
    res = minimize(tridFunc, x, jac=grad)
    print res
    print tridFunc(res['x'])
    #print x
