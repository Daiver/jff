import numpy as np
import cv2
import autograd.numpy as np
from autograd import grad, hessian


def nearestPointIndBrute(points, point):
    return np.argmin(points - point)

def drawFigure(canvas, points, color, translation=[0, 0]):
    for point in points:
        cv2.circle(canvas, 
                (int(point[0] + translation[0]), int(point[1] + translation[1])), 
                3, color, 3)

def createCanvas(width, height):
    return np.zeros((height, width, 3))

#x' = x*cosA - y*sinA
#y' = x*sinA + y*cosA

def transformPoint(translation, angle, point):
    angle = 0
    rotMat = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
        ])
    return translation + np.dot(rotMat, point)

def errorFunc(params, points, targets):
    angle = params[2]
    translation = params[0:2]
    res = 0
    for point, target in zip(points, targets):
        trans = transformPoint(translation, angle, point)
        diff = trans - target
        res += diff[0]*diff[0] + diff[1]*diff[1]
        #res += np.dot(diff, diff)
    return res

def gradientDescentStep(template, target, params, rate = 0.1):
    funcVal = errorFunc(params, template, target)
    gradVal = grad(lambda p: errorFunc(p, template, target))(params)
    print gradVal
    return params - rate*gradVal, funcVal

def simpleOpt(template, target):
    params = np.array([0, 0, 0.0], dtype=np.float32)
    nIters = 50

    currentTemplate = template
    for iter in xrange(nIters):
        indices = [nearestPointIndBrute(target, x) for x in currentTemplate]
        currentTarget = target[indices]
        params, err = gradientDescentStep(template, currentTarget, params, 0.1)
        print iter, err, params, indices
        translation = params[:2]
        params[2] = params[2] % (2 * np.pi)
        angle = params[2]
        currentTemplate = [transformPoint(translation, angle, x) for x in template]
        #print currentTemplate
        #print currentTarget
        canvas = createCanvas(1000, 500)
        drawFigure(canvas, target, (0, 255, 0), [0, 0])
        drawFigure(canvas, currentTemplate, (0, 0, 255))
        cv2.imshow('', canvas)
        #cv2.waitKey(100)
        cv2.waitKey(0)
    cv2.waitKey(0)

if __name__ == '__main__':

    target = np.array([
        [10, 10],
        [20, 10],
        [10, 20],
        [20, 20],
        [30, 10],
        [30, 20]
        ], dtype=np.float32) + [100, 100]

    template = np.array([
        [10, 10],
        [10, 20],
        [30, 10],
        [30, 20]
        ], dtype=np.float32) + [100, 0]

    simpleOpt(template, target)
    #target = transformPoints([0, 0], 3.14/4, target)

    #canvas = createCanvas(1000, 500)
    #drawFigure(canvas, target, (0, 255, 0), [0, 0])
    #drawFigure(canvas, template, (0, 0, 255))
    #cv2.imshow('', canvas)
    #cv2.waitKey()
