from loss import distFromPoint2LineSq, segmentLenConstraintSq, distFromPoint2PointSq
import autograd.numpy as np
from autograd import grad

from scipy.optimize import minimize

import cv2

def error(linePoints, points):
    res = 0
    p1 = linePoints[0, :]
    p2 = linePoints[1, :]
    for p in points:
        res += distFromPoint2LineSq(p, p1, p2)
    res += segmentLenConstraintSq(p1, p2, 1.0)
    res += distFromPoint2PointSq(p1, np.array([0.0, 0.0]))
    return res

def draw(linePoints, points):
    linePoints = np.copy(linePoints)
    points = np.copy(points)
    #assert False # not implemented
    canvasSize = (256, 256, 3)
    scale = 20
    offset = (5, 5)

    linePoints += offset
    linePoints *= scale

    p1 = linePoints[0]
    p2 = linePoints[1]

    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))

    canvas = np.zeros(canvasSize, dtype=np.uint8)
    print p1, p2
    #cv2.circle(canvas, p1, 2, (0, 255, 0), 2)
    #cv2.circle(canvas, p2, 2, (0, 255, 0), 2)
    cv2.line(canvas, p1, p2, (0, 255, 0))
    cv2.imshow('', canvas)
    cv2.waitKey()
    

def verboseError(linePoints, points):
    print linePoints
    draw(linePoints, points)
    return error(linePoints, points)

def visualExample01():
    import matplotlib.pyplot as plt

    linePoints = np.array([
            [0, 0],
            [1, 0]
    ], dtype=np.float32)

    points = np.array([
        [0, 1],
        [0, 2]
        ], dtype=np.float32)

    #draw(linePoints, points)
    f  = lambda x: error(x.reshape((-1, 2)), points)
    df = grad(f)
    fv = lambda x: verboseError(x.reshape(-1, 2), points)
    res = minimize(fv, linePoints.reshape(-1), jac=df, method='bfgs')
    print res
    print f(linePoints.reshape(-1))
    print df(linePoints.reshape(-1))
    

if __name__ == '__main__':
    visualExample01()

