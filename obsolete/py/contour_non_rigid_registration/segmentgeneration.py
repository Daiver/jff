import numpy as np


def mkSphericalyDistributedClosedSegments(nSegments):
    origin = np.array([0.0, 0.0], dtype=np.float32)
    points = []
    segments = []
    deltaAngle = (2*np.pi) / nSegments
    for i in xrange(nSegments):
        x = np.cos(deltaAngle * i)
        y = np.sin(deltaAngle * i)
        points.append((x, y))
        segments.append((i, (i + 1) % nSegments))
    return np.array(points), np.array(segments)

def splitExistingSegmentsByAverage(points, segmentsIndices):
    newPoints = []
    newSegmentIndices = []
    for segInd in segmentsIndices:
        s1, s2 = segInd
        p1, p2 = points[s1], points[s2]
        p3     = (p1 + p2) / 2.0
        s3     = len(points) + len(newPoints)
        newPoints.append(p3)
        newSegmentIndices.append((s1, s3))
        newSegmentIndices.append((s2, s3))
    return np.array(points.tolist() + newPoints), np.array(newSegmentIndices)

if __name__ == '__main__':
    import cv2
    import contourloss
    import drawtools
    nSegs = 50

    points, segmentsInds = mkSphericalyDistributedClosedSegments(nSegs)
    print points
    print segmentsInds

    canvasSize = (512, 512)
    scale = 200
    canvas = np.ones(canvasSize + (3,), dtype=np.uint8) * 255
    segments = contourloss.remapPointsToSegements(points, segmentsInds)
    drawtools.drawContour(canvas, points, segmentsInds, scale, drawCenterOfCoords=True)
    cv2.imshow('', canvas)
    cv2.waitKey()

    points, segmentsInds = splitExistingSegmentsByAverage(points, segmentsInds)
    drawtools.drawContour(canvas, points, segmentsInds, scale, drawCenterOfCoords=True)

    #drawtools.drawSegments(canvas, segments, scale=scale)
    #drawtools.drawPoints(canvas, points, scale=scale, drawCenterOfCoords=True)

    cv2.imshow('', canvas)
    cv2.waitKey()

