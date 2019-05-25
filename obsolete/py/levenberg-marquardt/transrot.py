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

#def rotateByQuat(quat, point):
def rotateByQuat(point, quat):
    return np.dot(((1.0/(np.dot(quat, quat))) * quatRotMatrix(quat)), point)

def func(point, w):
    quat, trans = w[:4], w[4:]
    return rotateByQuat(point, quat) + trans

def jacobi(point, w):
    #quat, trans = w[:4], w[4:]
    x, y, z = point
    return np.array([
        [ 0.0,  2*z, -2*y, 1, 0, 0],
        [-2*z,  0.0,  2*x, 0, 1, 0],
        [ 2*y, -2*x,  0.0, 0, 0, 1]
        ], dtype=np.float32)


def main():
    points   = np.array([
                [0, 1, 0],
                [2, 0, 0]
            ], dtype=np.float32)
    #trueQuat = np.array([3.0, 0.0, 1.0, 1.1], dtype=np.float32)
    myQuat   = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.1], dtype=np.float32)
    y2  = np.array([
        [-9, 5, 10],
        [-10, 3, 10]
#        [-1, -1, 30],
        #[0, 1, 30]
        #[0, 1, 0],
        #[2, 0, 0]
        ], dtype=np.float32).reshape((-1))
    y = np.copy(y2)
    print "Desire", y
    nIter = 2000
    err = 100000
    #for iter in xrange(nIter):
    iter = 0
    while err > 0.0001 and iter < nIter:
        y[:3] = y2[:3] - myQuat[4:]
        y[3:] = y2[3:] - myQuat[4:]
        nQuat = myQuat[:4] * [-1.0, -1, -1, 1]
        #f_x = applyFToMassPoints(func, myQuat, points)
        f_x = applyFToMassPoints(rotateByQuat, myQuat[:4], points)
        J = applyFToMassPoints(jacobi, None, f_x.reshape((-1, 3)))
        res = levenberg.lmIter(y, f_x, J, 0.1)
        myQuat += [res[0], res[1], res[2], 0, res[3], res[4], res[5]]
        f_x = applyFToMassPoints(func, myQuat, points)
        err = np.linalg.norm(f_x - y2)
        print iter, myQuat
        print err
        iter += 1

    print "Point", points
    print "Desire", y2
    f_x = applyFToMassPoints(func, myQuat, points)
    print "Res", f_x

if __name__ == '__main__':
    main()
#    quat = np.array([1.1, 0.2, 3.0, 1.0])
    #nQuat = quat * [-1, -1, -1, 1]
    #point = np.array([1.1, 2.2, -3.3])
    #transformed = rotateByQuat(point, quat)
    #print transformed
    #print point
    #print rotateByQuat(transformed, nQuat)
    #print nQuat
