import loss

import numpy as np

def findNearestSegmentForPoint(point, segments):
    assert len(segments) > 0
    bestInd = 0
    bestDist = np.finfo(np.float32)
    for ind, (p1, p2) in enumerate(segments):
        dist = loss.distFromPoint2SegmentSq(point, p1, p2)
        if dist < bestDist:
            bestDist = dist
            bestInd = ind
    return bestInd


def findNearestSegmentForPoints(points, segments):
    assert len(points) > 0
    res = [findNearestSegmentForPoint(p, segments) for p in points]
    return np.array(res)
