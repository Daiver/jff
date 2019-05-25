import numpy as np

def test01(A, x):
    #stepLength = 1.0
    stepLength = 0.1
    nIters = 30
    x = x / np.linalg.norm(x)
    for iter in xrange(nIters):
        grad = A.dot(x)
        projGrad = grad - x.dot(grad) * x
        step = x - projGrad * stepLength
        projStep = step / np.linalg.norm(step)
        x = projStep
        print iter, x.T.dot(A).dot(x), x
    return x

if __name__ == '__main__':
    A = np.array([
        [0, 0, 1],
        [1, 0.5, 1],
        [1, 0, 0]
        ])
    #A = A.T.dot(A)
    #print np.linalg.eigh(A)[0]
    #print np.linalg.eigh(A)[1]
    print np.linalg.eig(A)[0]
    print np.linalg.eig(A)[1]
    print test01(A, np.array([-1, -1, 1]))
    #print A.T.dot(A)

