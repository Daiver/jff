import numpy as np

def conjugateGradients(nIter, A, b, x0):
    x = np.array(x0, np.float32)
    r = b - A.dot(x)
    d = r
    for iter in xrange(nIter):
        alpha = r.T.dot(r)/(d.T.dot(A).dot(d))
        x1 = x + alpha * d
        r1 = r - alpha * A.dot(d)
        b1 = r1.T.dot(r1)/(r.T.dot(r))
        d1 = r1 + b1 * d

        d = d1
        r = r1
        x = x1

        err = sum((A.dot(x) - b)**2)
        print iter, err, x
    return x

if __name__ == '__main__':
    A = np.array([
        [3, 2],
        [2, 6]
        ])
    B = np.array([2, -8])
    print A.dot([2, -2]) - B
    #gradientDescentForQuad1(20, A, B, [-2, -2])
    conjugateGradients(20, A, B, [-2, -2])
