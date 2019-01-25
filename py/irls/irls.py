
import numpy as np
import matplotlib.pyplot as plt

def getLinearIRLSWeightsForP(p, diff):
    res = np.power(abs(diff) + 0.000001, p - 2)
    return res

def linearIRLS(p, nIter, X, Y):
    weights = np.ones(X.shape[0])
    beta    = np.ones(X.shape[1])
    for iter in xrange(nIter):
        W = np.diag(weights)
        A = X.T.dot(W).dot(X)
        B = X.T.dot(W).dot(Y)
        ans = np.linalg.lstsq(A, B)[0]
        diff = (Y - X.dot(ans))
        #print diff
        weights = getLinearIRLSWeightsForP(p, diff)
        #print weights
        #print ans
        print iter, 'err', np.linalg.norm(diff, p)
    print 'ans', ans
    return ans

if __name__ == '__main__':
    X = np.array([
        [3, 1],
        [4, 1],
        [5, 1],
        [7, 1],
        [9, 1],
        [3.5, 1],
        [1.5, 1],
        [0, 1]
        ])
    Y = np.array([
        2, 3, 4, 6.2, 7.9, 
        10, 9,
        -0.3
        ])
    sqAns  = linearIRLS(0.1, 10, X, Y)
    absAns = linearIRLS(1, 10, X, Y)
    lstans = linearIRLS(2, 10, X, Y)
    print np.linalg.lstsq(X, Y)[0]
    #print sum((Y - X.dot([ 0.51069767 , 3.34620155]))**2)
    xs = np.linspace(-2, 14)
    ys1 = [x*lstans[0] + lstans[1] for x in xs]
    ys2 = [x*absAns[0] + absAns[1] for x in xs]
    ys3 = [x*sqAns[0] + sqAns[1] for x in xs]
    #tmp = plt.plot(X[:, 0], Y, 'bo', xs, ys1, 'g-', xs, ys2, 'r', xs, ys3, 'y--')
    #print (tmp[0]).label
    tmp1, = plt.plot(X[:, 0], Y, 'bo', label="points")
    tmp2, = plt.plot(xs, ys1, 'g-', label="L2^2")
    tmp3, = plt.plot(xs, ys2, 'y--', label="L1")
    tmp4, = plt.plot(xs, ys3, 'r-', label="Lp, p = 0.1")
    plt.legend([tmp1, tmp2, tmp3, tmp4])
    #plt.legend(tmp)
    #print tmp
    plt.show()
