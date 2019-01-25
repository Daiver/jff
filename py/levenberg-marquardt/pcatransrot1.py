import numpy as np
import levenberg
from rot import applyFToMassPoints

#quat : [u, v, w, s]
def quatRotMatrix(quat):
    u, v, w, s = quat
    return np.array([
        [s**2 + u**2 - v**2 - w**2, 2 * (u*v - s*w),           2 * (u*w + s*v)          ],
        [2 * (u*v + s*w),           s**2 - u**2 + v**2 - w**2, 2 * (v*w - s*u)          ],
        [2 * (u*w - s*v),           2 * (v*w + s*u),           s**2 - u**2 - v**2 + w**2]
        ], dtype=np.float32)

def rotateByQuat(point, quat):
    return np.dot(((1.0/(np.dot(quat, quat))) * quatRotMatrix(quat)), point)

principialComponents = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.float32).T

def func(w, indexOfPoint):
    quat, weights = w[:4], w[4:]
    pc = principialComponents[3 * indexOfPoint:3 * indexOfPoint + 3, :]
    point = np.dot(pc, weights)
    return rotateByQuat(point, quat) #+ trans

def jacobiForPoint(w, indexOfPoint):
    quat, weights = w[:4], w[4:]#len(weights) == principialComponents.cols
    pc = principialComponents[3 * indexOfPoint : 3 * indexOfPoint + 3, :]
    x, y, z = rotateByQuat(np.dot(pc, weights), quat)
    return np.array([
        [ 0.0,  2*z, -2*y, pc[0, 0], pc[0, 1]],
        [-2*z,  0.0,  2*x, pc[1, 0], pc[1, 1]],
        [ 2*y, -2*x,  0.0, pc[2, 0], pc[2, 1]]
        ], dtype=np.float32)

def applyF(f, w):
    nPoints = principialComponents.shape[0] / 3
    res = []

    for i in xrange(nPoints):
        r = f(w, i)
        for x in r:
            res.append(x)

    return np.array(res, dtype=np.float32)

def main():
    w = np.array([0.0, 0.0, 0.0, 1.0, 0.300, 0.601], dtype=np.float32)
    angle = np.pi / 2.95
    vec = np.array([-1.0, 1.0, -0.0]) * np.sin(angle/2.0)
    quat  = np.array([vec[0], vec[1], vec[2], np.cos(angle/2.0)])
    y = np.array([
            [1, 0.5, 0.0],
            [0.5, 1, 0.0],
            [0, 0, 1.05],
            [1, 1, 1]
        ], dtype=np.float32)
    for i in xrange(len(y)):
        y[i] = rotateByQuat(y[i], quat)
    y = y.reshape((-1))
    
    f_x = applyF(func, w)
    err = np.linalg.norm(f_x - y)
    nIter = 1000
    iter = 0
    #for iter in xrange(nIter):
    while err > 0.001 and iter < nIter:
        J = applyF(jacobiForPoint, w)
        dW = levenberg.lmIter(y, f_x, J, 1.1)
        w += [dW[0], dW[1], dW[2], 0, dW[3], dW[4]]
        f_x = applyF(func, w)
        err = np.linalg.norm(f_x - y)
        print iter, err
        print w
        iter += 1
    print "End"
    print f_x
    print "Err", err
    print "Desire"
    print y

if __name__ == '__main__':
    main()

