import copy
import cv2
import numpy as np
import numpy.linalg as npli

from common import draw_arrow, checkDelaunay2
from Vertex import Vertex
from Face import Face
from HEdge import HEdge
from Mesh import Mesh

import test

def readPoints(fname):
    res = []
    with open(fname) as f:
        for s in f:
            if len(s) > 1:
                x, y = map(int, s.split())
                res.append((x, y))
    return res
    
scale = 1
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        points = readPoints(sys.argv[1])
        width = max(points, key=lambda (x, y): x)[0] + 0
        height = max(points, key=lambda (x, y): y)[1] + 0
        mesh = Mesh(width, height)
        for p in points:
            mesh.addPointAndTriangulate(p)
    else:
        mesh = Mesh(600, 600)

    print 'num of errors', test.countOfNotDelTriangles(mesh)

    def mouseClick(event, x, y, f, mesh):
        if event == cv2.EVENT_LBUTTONDOWN:
            print x, y
            point = (x/scale, (y)/scale)
            mesh.addPointAndTriangulate(point)
            cv2.imshow('2', mesh.draw())
            print test.countOfNotDelTriangles(mesh)
            #img1 = cv2.flip(mesh.draw(), 0)
            #points = mesh.faces[triIndx].vertexesPoints()
            #cv2.fillPoly(img1,np.array( [points]), (0,0,255))
            #cv2.imshow('423', cv2.flip(img1, 0))

    img2 = mesh.draw()
    cv2.imshow('2', img2)
    cv2.setMouseCallback('2', mouseClick, mesh)
    cv2.waitKey()

