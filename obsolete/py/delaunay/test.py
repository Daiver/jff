import cv2
import numpy as np
import numpy.linalg as npli

from common import draw_arrow, checkDelaunay2, computeCircleParams
from Vertex import Vertex
from Face import Face
from HEdge import HEdge
from Mesh import Mesh

import random

def countOfNotDelTriangles(mesh):
    err = 0
    img = mesh.draw()
    for face in mesh.faces:
        neigh = face.getNeighFaces()
        for fi, hei in neigh:
            points = face.getVertexesForDelTest(fi, hei)
            isDel = checkDelaunay2(points[0], points[1])
            if not isDel:
                a, b, c, d = computeCircleParams(points[0])
                exp = (points[1][0]**2 + points[1][1]**2)*a - points[1][0]*b + points[1][1]*c - d
                print '>>>', np.round(exp*np.sign(a), 3)
                hedge = mesh.hedges[hei]
                #print hei
                draw_arrow(img, hedge.opposite().head().point, hedge.head().point, (0,0,255))
                err += 1
                cv2.imwrite('dump.png', img)
    #cv2.imshow('fwfwef', img)
    return err

def bigRandomMeshTest():
    width, height = 900, 900
    numOfPoints = 2000
    mesh = Mesh(width, height)
    for i in xrange(numOfPoints):
        x = random.randint(0, width/5)*5
        y = random.randint(0, height/5)*5
        mesh.addPointAndTriangulate((x, y))
        if i % 10 == 0:
            if countOfNotDelTriangles(mesh) > 0:
                print 'ERROR', countOfNotDelTriangles(mesh), '>', x, y
                cv2.imshow('', mesh.draw())
                cv2.imwrite('dump.png', mesh.draw())
                with open('fail_vertexes_dump', 'w') as f:
                    for v in mesh.vertexes:
                        f.write('%d %d\n' % (v.point[0], v.point[1]))
                exit()
                cv2.waitKey(1000)
        if i % 50 == 0:
            print 'i', i
    #cv2.imshow('', mesh.draw())
    print 'End'
    return mesh

def testMeshAllFacesIsTriangles(mesh):
    print 'Testing triangles'
    for face in mesh.faces:
        e1 = face.edge()
        e2 = face.edge().next()
        e3 = face.edge().next().next()
        t1 = e1.index == e3.next().index
        t2 = e2.index != e1.index
        t3 = e1._head != e2._head and e3._head != e1._head
        t4 = e1._leftFace == e2._leftFace and e3._leftFace == e1._leftFace
        if not all([t1,t2, t3, t4]):
            print 'ERROR', t1, t2, t3, t4
    print 'End testing triangles'


if __name__ == '__main__':
    mesh = bigRandomMeshTest()
    testMeshAllFacesIsTriangles(mesh)
