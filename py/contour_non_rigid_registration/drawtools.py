import contourloss
import cv2
import numpy as np

def mkCanvas(canvasSize, fillColor=255):
    res = np.ones(canvasSize + (3,), dtype=np.uint8) * fillColor
    return res

def vert2Point(p): 
    return (int(p[0]), int(p[1]))

def canvasOffsetAsHalfOfCanvas(canvas):
    canvasSize = canvas.shape[:2]
    return (canvasSize[0] // 2, canvasSize[1] // 2)

def drawPoints(
        canvas, points, 
        scale = 1.0, offset = None, 
        color=(0, 255, 0),
        circleRadius=5,
        circleThickness=3,
        drawCenterOfCoords=False):
    if offset is None:
        offset = canvasOffsetAsHalfOfCanvas(canvas)
    points = points * scale + offset
    points = map(vert2Point, points)
    if drawCenterOfCoords:
        points += [offset]
    for p in points:
        cv2.circle(canvas, p, circleRadius, color, circleThickness)
    return canvas

def drawSegments(canvas, segments, scale=1.0, offset=None, color=(0, 255, 0)):
    if offset is None:
        offset = canvasOffsetAsHalfOfCanvas(canvas)
    segments = segments * scale + offset

    for p1, p2 in segments:
        p1 = vert2Point(p1)
        p2 = vert2Point(p2)
        cv2.line(canvas, p1, p2, color)

    return canvas

def drawContour(
        canvas, 
        points, segmentsInds, 
        scale=1.0, offset=None, 
        circleRadius=2, circleThickness=2,
        lineColor=(0, 255, 0),
        drawCenterOfCoords=False):
    if offset is None:
        offset = canvasOffsetAsHalfOfCanvas(canvas)
    points = points * scale + offset

    segments = contourloss.remapPointsToSegements(points, segmentsInds)

    for p1, p2 in segments:
	p1 = vert2Point(p1)
	p2 = vert2Point(p2)
	cv2.line(canvas, p1, p2, lineColor)

    for p in points.tolist() + ([offset] if drawCenterOfCoords else []):
	p = vert2Point(p)
	cv2.circle(canvas, p, circleRadius, (0, 255, 0), circleThickness)
    return canvas
