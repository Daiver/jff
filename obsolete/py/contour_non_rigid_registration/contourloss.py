from loss import *

def remapPointsToSegements(points, segmentsIndices):
    return points[segmentsIndices, :]

def pointsForLineLoss(segments, pointsForSegment, pointsForSegmentMap, pointsForSegmentWeights):
    res = 0.0
    assert len(pointsForSegment) == len(pointsForSegmentMap)
    xDiffs = segments[:, 1, 0] - segments[:, 0, 0]
    yDiffs = segments[:, 1, 1] - segments[:, 0, 1]
    hypoLengths = xDiffs**2 + yDiffs**2

    xDiffsSampled = xDiffs[pointsForSegmentMap]
    yDiffsSampled = yDiffs[pointsForSegmentMap]
    hypoLengthsSampled   = hypoLengths[pointsForSegmentMap]
    p1 = segments[pointsForSegmentMap, 0]
    p2 = segments[pointsForSegmentMap, 1]
    numerator = (
            (yDiffsSampled * pointsForSegment[:, 0]) 
          - (xDiffsSampled * pointsForSegment[:, 1])
          + p2[:, 0] * p1[:, 1]
          - p2[:, 1] * p1[:, 0]
    )**2
    losses = numerator / hypoLengthsSampled
    return np.dot(losses, pointsForSegmentWeights)

def pointsForPointLoss(points, pointsForPoints, pointsForPointMap):
    assert len(pointsForPointMap) == len(pointsForPoints)
    assert len(pointsForPointMap) == 0# Not implemented!
    return 0.0

def segmentsLengthEquallityAll2AllLoss(segments):
    nSegments = len(segments)
    p1 = segments[:, 0]
    p2 = segments[:, 1]
    lengths = np.sum((p2 - p1)**2, axis=1)
    x = y = np.arange(nSegments)
    cartessian = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    return np.sum((lengths[cartessian[:, 0]] - lengths[cartessian[:, 1]])**2)

def contourFittingLoss(
        points, segmentsInds,
        pointsForSegment, pointsForSegmentMap, pointsForSegmentWeights,
        pointsForPoints, pointsForPointMap,
        equalityWeight = 1000.0):
    
    if len(points.shape) == 1:
        points = points.reshape((-1, 2))
    segments = remapPointsToSegements(points, segmentsInds)

    res = 0.0
    res += pointsForLineLoss(segments, pointsForSegment, pointsForSegmentMap, pointsForSegmentWeights)
    res += pointsForPointLoss(points, pointsForPoints, pointsForPointMap)
    res += equalityWeight * segmentsLengthEquallityAll2AllLoss(segments)

    return res

    
