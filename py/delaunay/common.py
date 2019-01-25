import cv2
import numpy as np
import numpy.linalg as npli

def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=1, line_type=8, shift=0):
    cv2.line(image, p, q, color, thickness, line_type, shift)
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    cv2.line(image, p, q, color, thickness, line_type, shift)
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    cv2.line(image, p, q, color, thickness, line_type, shift)

def computeCircleParams(points):
    A = np.array([[points[0][0], points[0][1], 1], 
                  [points[1][0], points[1][1], 1], 
                  [points[2][0], points[2][1],1]])
    a = npli.det(A)
    B = np.array([
                  [points[0][0]**2 + points[0][1]**2, points[0][1], 1 ],
                  [points[1][0]**2 + points[1][1]**2, points[1][1], 1 ],
                  [points[2][0]**2 + points[2][1]**2, points[2][1], 1 ],
                  ])
    b = npli.det(B)
    C = np.array([
                  [points[0][0]**2 + points[0][1]**2, points[0][0], 1 ],
                  [points[1][0]**2 + points[1][1]**2, points[1][0], 1 ],
                  [points[2][0]**2 + points[2][1]**2, points[2][0], 1 ],
                  ])
    c = npli.det(C)
    D = np.array([
                  [points[0][0]**2 + points[0][1]**2, points[0][0], points[0][1] ],
                  [points[1][0]**2 + points[1][1]**2, points[1][0], points[1][1] ],
                  [points[2][0]**2 + points[2][1]**2, points[2][0], points[2][1] ],
                  ])
    d = npli.det(D)
    return a, b, c, d

def checkDelaunay2(points, p):
    a, b, c, d = computeCircleParams(points)
    exp = (p[0]**2 + p[1]**2)*a - p[0]*b + p[1]*c - d
    #print a, b, c, d
    return np.round(exp*np.sign(a), 3) >= 0

def checkDelaunayFromCachedParams(circleParams, p):
    a, b, c, d = circleParams
    exp = (p[0]**2 + p[1]**2)*a - p[0]*b + p[1]*c - d
    return exp*np.sign(a) >= 0

