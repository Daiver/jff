import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PolynomApproximator import PolynomApproximator
from SplitCurveEqualy import splitFunctionOnInterval

def heatMapToPointsAndWeights(heatMap):
    res = []
    weights = []
    threshold = 0.2
    for row in xrange(heatMap.shape[0]):
        for col in xrange(heatMap.shape[1]):
            val = heatMap[row, col]
            if val > threshold:
                res.append([col, row])
                weights.append(val)
    return np.array(res), np.array(weights)

def pyrUp(img, nIiterations=1):
    for _ in xrange(nIiterations):
        img = cv2.pyrUp(img)
    return img

if __name__ == '__main__':
    #imgName = 'imgs/0.png'
    #imgName = 'imgs/43.png'
    #imgName = 'imgs/51.png'
    #imgName = 'imgs/104.png'
    #imgName = 'imgs/115.png'
    imgName = 'imgs/187.png'
    #imgName = 'imgs/429.png'

    imgName = sys.argv[1] if len(sys.argv) > 1 else imgName

    img = cv2.imread(imgName).astype(np.float32) / 255.0
    points0, weights0 = heatMapToPointsAndWeights(img[:, :, 0])
    points1, weights1 = heatMapToPointsAndWeights(img[:, :, 1])
    

    power = 3

    poly0 = PolynomApproximator(points0, weights0, power)
    x0 = np.linspace(np.min(points0[:, 0]), np.max(points0[:, 0]), 50)
    y0 = poly0(x0)

    poly1 = PolynomApproximator(points1, weights1, power)
    x1 = np.linspace(np.min(points1[:, 0]), np.max(points1[:, 0]), 50)
    y1 = poly1(x1)

    print poly1.coeffs

    img = pyrUp(img, 1)
    img2 = np.copy(img)
    img2[img2 < 0.2] = 0
    imgSize = img.shape[0:2]
    chnl1 = np.zeros(imgSize, dtype=np.float32)
    chnl1[:, :] = img[:, :, 0]
    chnl2 = np.zeros(imgSize, dtype=np.float32)
    chnl2[:, :] = img[:, :, 1]

    #img2.fill(0)
    pts0 = (np.vstack([x0, y0]) * 2).round().astype(np.int32).T
    pts1 = (np.vstack([x1, y1]) * 2).round().astype(np.int32).T
    cv2.polylines(img2, [pts0.reshape((-1, 1, 2))], False, (0, 0, 1.0))
    cv2.polylines(img2, [pts1.reshape((-1, 1, 2))], False, (0, 1, 1.0))

    sampledPoints0 = splitFunctionOnInterval(poly0, np.min(points0[:, 0]), np.max(points0[:, 0]), 200, 30)
    x2 = sampledPoints0[:, 0]
    y2 = sampledPoints0[:, 1]
    pts2 = (np.vstack([x2, y2]) * 2).round().astype(np.int32).T
    for p in pts2:
        cv2.circle(img2, (int(p[0]), int(p[1])), 2, (0, 1, 0))

    product = chnl1 * chnl2
    cv2.imshow('product', product)

    cv2.imshow('', img)
    cv2.imshow('img2', img2)
    cv2.imshow('1', chnl1)
    cv2.imshow('2', chnl2)#'''
    cv2.waitKey()
   

    #plt.plot(points0[:, 0], points0[:, 1], 'ro')
    #plt.plot(x0, y0, 'r-')
    #plt.plot(pts0[:, 0], pts0[:, 1], 'r-')
    #plt.plot(points1[:, 0], points1[:, 1], 'go')
    #plt.plot(x1, y1, 'g-')
    plt.show()

