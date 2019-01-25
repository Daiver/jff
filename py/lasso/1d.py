from softthresholding import softthresholding
import numpy as np
import cvxpy


def run1dCustom(b, clambda):
    return softthresholding(b, clambda)

def run1dCVXPy(b, clambda):
    x = cvxpy.Variable()
    obj = cvxpy.Minimize(clambda*cvxpy.abs(x) + 0.5*cvxpy.square(x - b))
    prob = cvxpy.Problem(obj, [])
    prob.solve(verbose=False)
    return x.value

def test01():
    for iter in xrange(1000):
        b = np.random.uniform(-5, 5)
        clambda = np.random.uniform(0, 10)
        res = run1dCustom(b, clambda)
        ans = run1dCVXPy(b, clambda)
        if np.abs(res - ans) > 1e-3:
            print 'wrong?', b, clambda, res, ans

if __name__ == '__main__':
    test01()
