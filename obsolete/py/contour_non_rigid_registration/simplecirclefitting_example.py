import loss
import contourloss
import drawtools
import segmentgeneration
import searchnearestsegment

from autograd import grad
import autograd.numpy as np
np.random.seed(42)
from scipy.optimize import minimize

import cv2

import cProfile

def drawScene(linePoints, segsInds, targetPoints):
    canvasSize = (800, 800)
    scale = 170.0
    canvas = drawtools.mkCanvas(canvasSize)
    drawtools.drawPoints(canvas, targetPoints, scale=scale, circleRadius=2, circleThickness=2, color=(0, 0, 255))
    drawtools.drawContour(canvas, linePoints, segsInds, scale=scale, circleRadius=2, circleThickness=2)

    segments = contourloss.remapPointsToSegements(linePoints, segsInds)
    targetPoints2SegmentsMap = searchnearestsegment.findNearestSegmentForPoints(
            targetPoints, segments)
    corrsLines = correspondeceLines(segments, targetPoints, targetPoints2SegmentsMap)
    drawtools.drawSegments(canvas, corrsLines, scale, color=(255, 0, 0))

    cv2.imshow('target', canvas)
    cv2.waitKey(50)

def mkErrorFunction(linePoints, segsInds, targetPoints):
    segments = contourloss.remapPointsToSegements(linePoints, segsInds)
    targetPoints2SegmentsMap = searchnearestsegment.findNearestSegmentForPoints(targetPoints, segments)
    targetPoints2SegmentsWeights = np.ones(len(targetPoints), dtype=np.float32)

    eqWeight = 0.1 * len(targetPoints) * float(len(segsInds))
    print 'eqWeight', eqWeight

    def loss(linePoints):
        return contourloss.contourFittingLoss(
            linePoints, segsInds, 
            targetPoints, targetPoints2SegmentsMap, targetPoints2SegmentsWeights,
            [], [], eqWeight)
    return loss

def correspondeceLines(segments, targetPoints, targetPoints2SegmentsMap):
    return np.array([
        (point, loss.projectPoint2Segment(point, seg[0], seg[1]))
        for point, seg in zip(targetPoints, segments[targetPoints2SegmentsMap])
    ], dtype=np.float32)

def main1():
    targetPoints, _ = segmentgeneration.mkSphericalyDistributedClosedSegments(300)
    targetPoints *= np.array([2.0, 1.2])

    nSegments = 64
    linePoints, segsInds = segmentgeneration.mkSphericalyDistributedClosedSegments(nSegments)

    f = mkErrorFunction(linePoints, segsInds, targetPoints)
    #df = None
    df = grad(f)
    res = minimize(f, linePoints.reshape(-1), jac=df, method='bfgs')
    print res

    drawScene(linePoints, segsInds, targetPoints)
    linePoints = res['x'].reshape((-1, 2))
    drawScene(linePoints, segsInds, targetPoints)
    segments = contourloss.remapPointsToSegements(linePoints, segsInds)

def main2():
    targetPoints, _ = segmentgeneration.mkSphericalyDistributedClosedSegments(600)
    targetPoints *= np.array([2.0, 1.2])

    targetPoints += np.random.normal(size=targetPoints.shape, scale=0.02)

    nSegments = 8
    linePoints, segsInds = segmentgeneration.mkSphericalyDistributedClosedSegments(nSegments)

    drawScene(linePoints, segsInds, targetPoints)
    nSubdivisions = 5
    nICPIters = 10
    nSolverIters = 20
    for subdivisionIter in range(nSubdivisions):
        if subdivisionIter > 0:
            segments = contourloss.remapPointsToSegements(linePoints, segsInds)
            eqErr = contourloss.segmentsLengthEquallityAll2AllLoss(segments)
            print 'eqErr before subdiv', eqErr
            linePoints, segsInds = segmentgeneration.splitExistingSegmentsByAverage(linePoints, segsInds)
            segments = contourloss.remapPointsToSegements(linePoints, segsInds)
            eqErr = contourloss.segmentsLengthEquallityAll2AllLoss(segments)
            print 'eqErr after subdiv', eqErr
            drawScene(linePoints, segsInds, targetPoints)
        for icpIter in range(nICPIters):
            f = mkErrorFunction(linePoints, segsInds, targetPoints)
            #df = None
            df = grad(f)
            res = minimize(f, linePoints.reshape(-1), jac=df, method='bfgs', options={'maxiter': nSolverIters})
            print icpIter, '/', nICPIters
            print res['nfev'], res['nit'], res['njev'], res['fun']

            linePoints = res['x'].reshape((-1, 2))
            drawScene(linePoints, segsInds, targetPoints)
            segments = contourloss.remapPointsToSegements(linePoints, segsInds)
            eqErr = contourloss.segmentsLengthEquallityAll2AllLoss(segments)
            print 'eqErr', eqErr
    cv2.waitKey()

if __name__ == '__main__':
    #main1()
    main2()

