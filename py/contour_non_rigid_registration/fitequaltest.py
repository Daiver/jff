from loss import distFromPoint2LineSq, segmentLenConstraintSq, distFromPoint2PointSq, segmentsEqualLengthConstraint
import autograd.numpy as np
from autograd import grad

from scipy.optimize import minimize

import cv2

def error(linePoints, points2Fit):
    p1 = linePoints[0]
    p2 = linePoints[1]
    p3 = linePoints[2]

    constrWeight = 10.0

    res = 0
    res += distFromPoint2PointSq(p1, points2Fit[0])
    res += distFromPoint2PointSq(p3, points2Fit[1])
    res += constrWeight*segmentsEqualLengthConstraint(p1, p2, p2, p3)
    #res += segmentsEqualLengthConstraint(p2, p3)

    return res

def draw(linePoints, points2Fit):
    linePoints = np.copy(linePoints)
    points2Fit = np.copy(points2Fit)
    #assert False # not implemented
    canvasSize = (256, 256, 3)
    scale = 20
    offset = (5, 5)

    points2Fit += offset
    points2Fit *= scale

    linePoints += offset
    linePoints *= scale

    p1 = linePoints[0]
    p2 = linePoints[1]
    p3 = linePoints[2]

    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    p3 = (int(p3[0]), int(p3[1]))

    canvas = np.zeros(canvasSize, dtype=np.uint8)
    print p1, p2
    #cv2.circle(canvas, p1, 2, (0, 255, 0), 2)
    #cv2.circle(canvas, p2, 2, (0, 255, 0), 2)
    cv2.line(canvas, p1, p2, (0, 255, 0))
    cv2.line(canvas, p2, p3, (0, 255, 0))
    cv2.circle(canvas, p1, 2, (0, 255, 0))
    cv2.circle(canvas, p2, 2, (0, 255, 0))
    cv2.circle(canvas, p3, 2, (0, 255, 0))

    p1 = points2Fit[0]
    p2 = points2Fit[1]
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    cv2.circle(canvas, p1, 3, (0, 0, 255), 3)
    cv2.circle(canvas, p2, 3, (0, 0, 255), 3)

    cv2.imshow('', canvas)
    cv2.waitKey()

def verboseError(*args):
    draw(*args)
    return error(*args)

def visualTest(linePoints, points2Fit):
    f  = lambda x: error(x.reshape((-1, 2)), points2Fit)
    fv = lambda x: verboseError(x.reshape((-1, 2)), points2Fit)
    df = grad(f)
    #fv = lambda x: verboseError(x.reshape(-1, 2), points)
    print f(linePoints.reshape(-1))
    print df(linePoints.reshape(-1))
    #res = minimize(f, linePoints.reshape(-1), jac=df, method='bfgs')
    res = minimize(fv, linePoints.reshape(-1), jac=df, method='bfgs')
    print res
    newLines = res['x'].reshape((-1, 2))
    print distFromPoint2PointSq(newLines[0], newLines[1]), distFromPoint2PointSq(newLines[1], newLines[2])

def visualTest01():
    linePoints = np.array([
        [ 0, -1],
        [ 0,  0],
        [ 0,  1]
    ], dtype=np.float32)

    points2Fit = np.array([
        [-2, 0],
        [ 2, 0]
    ], dtype=np.float32)

    visualTest(linePoints, points2Fit) 

def visualTest02():
    linePoints = np.array([
        [ 0, -1],
        [ 0,  0],
        [ 0,  1]
    ], dtype=np.float32)

    points2Fit = np.array([
        [-2, 0],
        [ 0, 2]
    ], dtype=np.float32)

    visualTest(linePoints, points2Fit) 



if __name__ == '__main__':
    visualTest01()
    visualTest02()
    
