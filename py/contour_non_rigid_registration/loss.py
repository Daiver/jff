import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad


def distFromPoint2LineSq(point, p1, p2):
    yDiff = p2[1] - p1[1]
    xDiff = p2[0] - p1[0]

    hypoLen = xDiff ** 2 + yDiff ** 2

    numerator = yDiff * point[0] - xDiff * point[1] + p2[0] * p1[1] - p2[1]*p1[0]
    return (numerator**2) / hypoLen

def distFromPoint2PointSq(point1, point2):
    diff = point1 - point2
    return np.dot(diff, diff)

def segmentLenConstraintSq(p1, p2, length):
    lengthSq = length * length
    segLen = distFromPoint2PointSq(p1, p2)
    return (segLen - lengthSq) ** 2

def segmentsEqualLengthConstraint(p1, p2, p3, p4):
    d1 = distFromPoint2PointSq(p1, p2)
    d2 = distFromPoint2PointSq(p3, p4)
    return (d1 - d2) ** 2

def projectPoint2Segment(point, p1, p2):
    l2 = distFromPoint2PointSq(p1, p2)
    t = max(0.0, min(1.0, np.dot(point - p1, p2 - p1) / l2))
    return p1 + t * (p2 - p1)

def distFromPoint2SegmentSq(point, p1, p2):
    projection = projectPoint2Segment(point, p1, p2)
    return distFromPoint2PointSq(point, projection)

if __name__ == '__main__':
    point = np.array([1, 1], dtype=np.float32)
    p1    = np.array([0, 0], dtype=np.float32)
    p2    = np.array([1, 0], dtype=np.float32)

    print grad(distFromPoint2LineSq, argnum=0)(point, p1, p2)
    print grad(distFromPoint2LineSq, argnum=1)(point, p1, p2)
    print grad(distFromPoint2LineSq, argnum=2)(point, p1, p2)


