import numpy as np
np.set_printoptions(edgeitems=50, linewidth=175)
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.optimize import minimize

#TODO: SUPER SLOW SWITCH TO C++
def takePixelInterpolated(img, pt):
    rows, cols = img.shape
    y, x = pt

    x0 = cv2.borderInterpolate(int((x)),   cols, cv2.BORDER_REFLECT_101);
    x1 = cv2.borderInterpolate(int((x+1)), cols, cv2.BORDER_REFLECT_101);
    y0 = cv2.borderInterpolate(int((y)),   rows, cv2.BORDER_REFLECT_101);
    y1 = cv2.borderInterpolate(int((y+1)), rows, cv2.BORDER_REFLECT_101);

    a = x - int(x)
    c = y - int(y)

    #vx0y0 = 
    return (img[y0, x0] * (1.0 - a) + img[y0, x1] * a) * (1.0 - c) + (img[y1, x0] * (1.0 - a) + img[y1, x1] * a) * c

#TODO: USE CONSTANT FOLDING
def discretPoints2ContourLoss(points, contourImg):
    #distField = cv2.distanceTransform(img, cv2.DIST_L1, 0)
    distField = cv2.distanceTransform(contourImg, cv2.DIST_L2, 0)
    return sum(takePixelInterpolated(distField, p) for p in points)
    #return sum(distField[p[0], p[1]] for p in points.round().astype(np.uint32))

#TODO: USE CONSTANT FOLDING
def discretPoints2ContourGrad(points, contourImg):
    distField = cv2.distanceTransform(contourImg, cv2.DIST_L2, 0)
    dxKernel = np.array([
        [ 0, 0, 0],
        [-1, 0, 1],
        [ 0, 0, 0]
    ], dtype=np.float32) / 2.0
    dyKernel = np.array([
        [0, -1,  0],
        [0,  0,  0],
        [0,  1,  0]
    ], dtype=np.float32) / 2.0

    #dxKernel = np.array([[]], dtype=np.float32)

    dx = cv2.filter2D(distField, -1, dxKernel)
    dy = cv2.filter2D(distField, -1, dyKernel)
    
    res = np.zeros_like(points)
    for i, p in enumerate(points):
        res[i, 0] = takePixelInterpolated(dx, p)
        res[i, 1] = takePixelInterpolated(dy, p)
    print('p>', points)
    print('g>', res)
    return res

def main():
    img = np.zeros((7, 7), dtype=np.uint8)
    img[:] = 1
    img[2, 2] = 0
    img[3, 3] = 0
    img[4, 3] = 0
    print(img)

    points = np.array([
        [4, 2], 
    ])

    print('dist', discretPoints2ContourLoss(points, img)) 

    loss = lambda x: discretPoints2ContourLoss(x.reshape((-1, 2)), img)
    def numJac(x):
        print('x>', x)
        #AGAIN, SUPER SLOW
        dx = 0.3
        #dx = 0.51
        #dx = 1.0
        res = np.zeros(x.shape)
        fx = loss(x)
        for i in range(x.shape[0]):
            x1 = np.copy(x)
            x1[i] -= dx
            fx1 = loss(x1)
            x1[i] += 2*dx
            fx2 = loss(x1)
            res[i] = (fx2 - fx1) / dx / 2.0
            #print(fx1, fx2, x1)
        print('g>', res)
        return res

    def discretJac(x):
        return discretPoints2ContourGrad(x.reshape((-1, 2)), img).reshape(-1)

    jac = numJac
    #jac = None
    #jac = discretJac

    bounds = None

    optRes = minimize(loss, points.reshape(-1), jac=jac, bounds=bounds)
    print(optRes)
    print('x>', optRes['x'].round())
    


if __name__ == '__main__':
    #main() 
    img = np.zeros((7, 7), dtype=np.uint8)
    img[:] = 1
    img[3, 3] = 0
    img[4, 3] = 0
    print(img)
    distField = cv2.distanceTransform(img, cv2.DIST_L1, 0)
    print(distField)

    for i in range(11):
        #point = np.array([4 + 0.1 * i, 2])
        point = np.array([4, 2 + 0.1 * i])
        pixVal = takePixelInterpolated(distField, point)
        print(pixVal, point)

