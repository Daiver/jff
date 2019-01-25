import cv2
import numpy as np

def applyTransform(transformation, point, scale=1):
    return (point + transformation)*scale

def inverseTransform(transformation, point):
    return point - transformation

def isPointInsidePicture(shape, point):
    return point[0] >= 0 and point[1] >= 0 and point[0] < shape[1] and point[1] < shape[0]

def computeDifferences(img, pairs):
    res = np.zeros(len(pairs), dtype=np.float32)
    for i, (p1, p2) in enumerate(pairs):
        #val1 = 

        try:
            val1 = img[int(p1[1]),int(p1[0])]
        except IndexError:
            val1 = 0

        try:
            val2 = img[int(p2[1]),int(p2[0])]
        except IndexError:
            val2 = 0

#        if isPointInsidePicture(img.shape, p1):
            #val1 = img[int(p1[1]), int(p1[0])] 
        #else:
            #val1 = 0

        #if isPointInsidePicture(img.shape, p2):
            #val2 = img[int(p2[1]), int(p2[0])] 
        #else:
            #val2 = 0


        #val2 = img[int(p2[1]), int(p2[0])] if isPointInsidePicture(img.shape, p2) else 0
        res[i] = float(val1) - float(val2)
        #print val1, val2, res[i]
    return res


def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=1, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html
    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow 
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head 
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head 
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)

def parzenWindow(data, width):
    pass

