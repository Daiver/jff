import sys
import numpy as np
from PolynomApproximator import PolynomApproximator
from SplitCurveEqualy import splitFunctionOnInterval

def heatMapToPointsAndWeights(heatMap, threshold = 0.2):
    res = []
    weights = []
    for row in xrange(heatMap.shape[0]):
        for col in xrange(heatMap.shape[1]):
            val = heatMap[row, col]
            if val > threshold:
                res.append([col, row])
                weights.append(val)
    return np.array(res), np.array(weights)

def heatMapToEyelidPoints(heatMap):
    polyPower = 3
    heatMapThreshold = 0.2

    nPointsForIntegration = 200

    nSplitPoints = 30

    points, weights = heatMapToPointsAndWeights(heatMap)
    poly = PolynomApproximator(points, weights, polyPower)

    sampledPoints = splitFunctionOnInterval(
            poly, 
            np.min(points[:, 0]), np.max(points[:, 0]), 
            nPointsForIntegration, nSplitPoints)

    return sampledPoints
