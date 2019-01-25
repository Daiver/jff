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

def jacobiFromRotationWithQi(point, w):
    x, y, z = point
    return np.array([
        [ 0.0,  2*z, -2*y],
        [-2*z,  0.0,  2*x],
        [ 2*y, -2*x,  0.0]
        ], dtype=np.float32)


def main():
    points   = np.array([
                [0, 1, 0],
                [2, 0, 0]
            ], dtype=np.float32)
    #trueQuat = np.array([3.0, 0.0, 1.0, 1.1], dtype=np.float32)
    myQuat   = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    #y = applyFToMassPoints(rotateByQuat, trueQuat, points)
    #QVector3D(2.62105, 1.55263, 0.468421)
    y   = np.array([
        #[1, 0, 0],
        #[0, -2, 0]
        [-1, 0, 0],
        [0, 2, 0]
        ], dtype=np.float32).reshape((-1))
    print "Desire", y
    nIter = 200
    err = 100000
    #for iter in xrange(nIter):
    iter = 0
    while err > 0.001 and iter < 1000:
        f_x = applyFToMassPoints(rotateByQuat, myQuat, points)
        #f_x = rotateByQuat(myQuat, point)
        #J = jacobiFromRotationWithQi(f_x)
        J = applyFToMassPoints(jacobiFromRotationWithQi, myQuat, f_x.reshape((-1, 3)))
        res = levenberg.lmIter(y, f_x, J, 0.1)
        myQuat += [res[0], res[1], res[2], 0]
        #myQuat[:3] /= np.sqrt(1.0 + sum([x**2 for x in myQuat[:3]]))
        f_x = applyFToMassPoints(rotateByQuat, myQuat, points)
        err = np.linalg.norm(f_x - y)
        print iter, myQuat, err
        iter += 1

    print "Point", points
    print "Desire", y
    f_x = applyFToMassPoints(rotateByQuat, myQuat, points)
    print "Res", f_x


def main1():
    point    = np.array([-7.0, 2.5, 1.5], dtype=np.float32)
    trueQuat = np.array([-3.0, -7.0, 1.0, 1.0], dtype=np.float32)
    myQuat   = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    y = rotateByQuat(trueQuat, point)
    #QVector3D(2.62105, 1.55263, 0.468421)
    print "Desire", y
    nIter = 200
    err = 100000
    #for iter in xrange(nIter):
    iter = 0
    while err > 0.001 and iter < 1000:
        f_x = rotateByQuat(myQuat, point)
        J = jacobiFromRotationWithQi(f_x)
        res = levenberg.lmIter(y, f_x, J, 0.1)
        myQuat += [res[0], res[1], res[2], 0]
        myQuat[:3] /= np.sqrt(1.0 + sum([x**2 for x in myQuat[:3]]))
        err = np.linalg.norm(rotateByQuat(myQuat, point) - y)
        print iter, myQuat, err
        iter += 1

    print "Point", point
    print "Desire", rotateByQuat(trueQuat, point)
    print "Res", rotateByQuat(myQuat, point)

if __name__ == '__main__':
    main()
