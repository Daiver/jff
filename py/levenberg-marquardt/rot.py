import levenberg
import numpy as np

#x: x, y, z
#w vx vy vz theta
#w : theta
def func(x, w):
    x = np.array(x)
    u = np.array([0, 0, 1])
    theta = w[0]
    a = x * np.cos(theta)
    b = np.cross(u, x) * np.sin(theta)
    c = u * (np.dot(u, x)) * (1.0 - np.cos(theta))
    #print a
    #print b
    #print c
    return a + b + c

def grad(x, w):
    x = np.array(x)
    u = np.array([0, 0, 1])
    theta = w[0]
    a = -x * np.sin(theta)
    b = np.cross(u, x) * np.cos(theta)
    c = u * np.dot(u, x) * np.sin(theta)
    return np.array([a + b + c]).reshape((3, 1))

def applyFToMassPoints(f, w, points):
    arr = [] 
    for p in points:
        res = f(p, w)
        for i in xrange(len(res)):
            arr.append(res[i])
    return np.array(arr)

def main1():
    point = np.array([1, 1, 1.0])
    angle = 3.14/8
    desire = func(point, np.array([angle]))
    print 'Des', desire
    w = np.array([-4.01])
    for iter in xrange(10):
        f_x = func(point, w)
        J = grad(point, w)
        w += levenberg.lmIter(desire, f_x, J)
        print f_x
        print w
    print "Err", sum(abs((func(point, w) - desire)))

if __name__ == '__main__':
    points = [
                np.array([1, 1, 1.0]),
                np.array([2, 2, 1.0])
            ]
    angle = 3.14/8
    desire = applyFToMassPoints(func, np.array([angle]), points)
    desire += np.random.rand(len(desire)) * 0.05
    print 'Des', desire
    w = np.array([-2.01])
    for iter in xrange(10):
        f_x = applyFToMassPoints(func, w, points)
        J = applyFToMassPoints(grad, w, points)
        #J = grad(point, w)
        w += levenberg.lmIter(desire, f_x, J)
        print f_x
        print w
    print 'Real angle', angle
    print 'res', w
    print w[0] % (2 * np.pi)
    #print "Err", sum(abs((func(point, w) - desire)))


