import copy
import numpy as np
import cv2

from Face import Face
from Vertex import Vertex
from HEdge import HEdge
import common
from common import draw_arrow, checkDelaunay2

class Mesh:
    def __init__(self, width, height):
        self.vertexes = []
        self.faces    = []
        self.hedges   = []
        self.numOfVertex = 0
        self.numOfFaces  = 0
        self.numOfHEdges = 0

        self.width = width
        self.height = height
        self.newVertex((0, 0), 0)
        self.newVertex((width, 0), 1)
        self.newVertex((width, height), 2)
        self.newVertex((0, height), 3)

        self.newFace(0)
        self.newHEdge(1, 1, 2, 0) #0
        self.newHEdge(0, 0, 5, None)#1
        self.newHEdge(3, 3, 4, 0)#2
        self.newHEdge(1, 2, 6, 1)#3
        self.newHEdge(0, 5, 0, 0) #4
        self.newHEdge(3, 4, 3, None) #5

        self.newFace(6)
        self.newHEdge(2, 7, 8, 1) #6
        self.newHEdge(1, 6, 1, None) #7
        self.newHEdge(3, 9, 3, 1) #8
        self.newHEdge(2, 8, 7, None) #9

        for f in self.faces:
            f.calculateCircleParams()


    def newVertex(self, point, _edge):
        self.numOfVertex += 1
        self.vertexes.append(Vertex(self.vertexes, self.faces, self.hedges, point, _edge))
        self.vertexes[-1].index = self.numOfVertex - 1
        return len(self.vertexes) - 1

    def newFace(self, _edge):
        self.numOfFaces += 1
        self.faces.append(Face(self.vertexes, self.faces, self.hedges, _edge))
        self.faces[-1].index = self.numOfFaces - 1
        return len(self.faces) - 1

    def newHEdge(self, _head, _opposite, _next, _leftFace=None):
        self.numOfHEdges += 1
        self.hedges.append(
                HEdge(self.vertexes, self.faces, self.hedges, 
                    _head, _opposite, _next, _leftFace))
        self.hedges[-1].index = self.numOfHEdges - 1
        return len(self.hedges) - 1

    def drawWithoutFirstPoints(self, startIndex):
        canvas = np.ones((self.height, self.width, 3))*255
        scale = 1

        for v in self.vertexes:
            if v.index < startIndex: continue
            cv2.circle(canvas, (v.point[0]*scale, v.point[1]*scale), 3, (0, 255, 0), 3)

        stHEdge = self.hedges[0]
        visited = [False for _ in self.hedges]
        stack = [stHEdge]
        while len(stack) > 0:
            curHEdge = stack.pop()
            if visited[curHEdge.index]: continue
            visited[curHEdge.index] = True
            p1, p2 = curHEdge.head().point, curHEdge.opposite().head().point
            '''p3 = ((p1[0] + p2[0])/2.0, (p1[1] + p2[1])/2.0)
            a = 1.0/(p1[0] - p2[0] + 0.001)
            b = -1.0/(p1[1] - p2[1] + 0.001)
            x4 = int(p3[0]+1)
            p4 = (x4, int(b/a*(x4 - p3[0]) + p3[1]))

            cv2.line(canvas, 
                    (p4[0]*scale, p4[1]*scale), 
                    (int(p3[0]*scale), int(p3[1]*scale)), (255, 0, 255), 3)'''

            #cv2.line(canvas, 
            #        (p1[0]*scale, p1[1]*scale), (p2[0]*scale, p2[1]*scale), (255, 0, 0), 3)
            if (curHEdge._head >= startIndex) and (curHEdge.opposite()._head >= startIndex):
                draw_arrow(canvas, 
                    (p2[0]*scale, p2[1]*scale), (p1[0]*scale, p1[1]*scale), (255, 0, 0))
            stack += [curHEdge.opposite(), curHEdge.next()]
            #cv2.imshow('', cv2.flip(canvas, 0))
            #cv2.waitKey()

        return canvas
        #return  cv2.flip(canvas, 0)

    def draw(self):
        return self.drawWithoutFirstPoints(0)

    def flip(self, badHEdgeInd):#checkIt
        badHEdge   = self.hedges[badHEdgeInd]
        badHEdgeOp = self.hedges[badHEdgeInd].opposite()
        badHEdge.leftFace()._edge = badHEdge.index
        badHEdgeOp.leftFace()._edge = badHEdgeOp.index
        #print 'badHEdgeOp', badHEdgeOp.next()
        cpBadHEdge   = copy.copy(badHEdge)         
        cpBadHEdgeOp = copy.copy(badHEdgeOp)         

        #change 
        badHEdge._head = cpBadHEdge.next()._head
        badHEdge._next = cpBadHEdge.next()._next

        #print 'badHEdgeOp', cpBadHEdgeOp.next()
        badHEdgeOp._head = cpBadHEdgeOp.next()._head
        badHEdgeOp._next = cpBadHEdgeOp.next()._next

        cpBadHEdge.next().next()._next = cpBadHEdgeOp._next 
        cpBadHEdgeOp.next()._leftFace = badHEdge._leftFace
 
        cpBadHEdgeOp.next().next()._next = cpBadHEdge._next 
        cpBadHEdge.next()._leftFace = badHEdgeOp._leftFace

        cpBadHEdge.next()._next = badHEdgeOp.index
        cpBadHEdgeOp.next()._next = badHEdge.index

        badHEdge.leftFace().calculateCircleParams()
        badHEdgeOp.leftFace().calculateCircleParams()

    def addPointInsideTriangle(self, faceIdx, p):
        newHEdgeIdx = len(self.hedges) 
        newFaceIdx = len(self.faces)
        newVertexIdx = self.newVertex(p, newHEdgeIdx)

        face = self.faces[faceIdx]
        stHEdge = face.edge()

        #newhedgeidx
        #he0 = [stHEdge.opposite()._head, newHEdgeIdx + 1, stHEdge.index,        faceIdx]
        self.newHEdge(stHEdge.opposite()._head, newHEdgeIdx + 1, stHEdge.index, faceIdx)
        #then is sthedge
        #newhedgeidx + 1
        #he1 = [newVertexIdx, newHEdgeIdx,newHEdgeIdx + 5,      newFaceIdx ]
        self.newHEdge(newVertexIdx, newHEdgeIdx,newHEdgeIdx + 5, newFaceIdx)
        #stHEdge.next().next()._head = he2.index
        # newhedgeidx + 2
        #he2 = [newVertexIdx,             newHEdgeIdx + 3, newHEdgeIdx,          faceIdx]
        self.newHEdge(newVertexIdx, newHEdgeIdx + 3, newHEdgeIdx,faceIdx)
        #stHEdge._next = newhedgeidx + 2
        # newvertexidx + 3
        #he3 = [stHEdge._head,            newHEdgeIdx + 2, stHEdge._next,        newFaceIdx + 1] 
        self.newHEdge(stHEdge._head,newHEdgeIdx + 2, stHEdge._next, newFaceIdx + 1)
        # newVertexIdx + 4
        #he4 = [newVertexIdx,             newHEdgeIdx + 5, newHEdgeIdx + 3,      newFaceIdx + 1] 
        self.newHEdge(newVertexIdx, newHEdgeIdx + 5, newHEdgeIdx + 3,newFaceIdx + 1)
        # newVertexIdx + 5
        #he5 = [stHEdge.next()._head,     newHEdgeIdx + 4, stHEdge.next()._next, newFaceIdx]
        self.newHEdge(stHEdge.next()._head,     newHEdgeIdx + 4, stHEdge.next()._next, newFaceIdx)

        f1 = self.newFace(newHEdgeIdx + 1)
        f2 = self.newFace(newHEdgeIdx + 3)

        stHEdge.next().next()._next = newHEdgeIdx + 1
        stHEdge.next().next()._leftFace = newFaceIdx 
        stHEdge.next()._next = newHEdgeIdx + 4
        stHEdge.next()._leftFace = newFaceIdx + 1

        stHEdge._next = newHEdgeIdx + 2

        res = faceIdx, f1, f2
        for f in res:
            self.faces[f].calculateCircleParams()
        return res

    def findTriangleByPoint(self, p):
        for face in self.faces:
            if face.isPointBelong(p):
                return face.index

    def triangulateAll(self):
        for face in self.faces:
            neigh = face.getNeighFaces()
            for fi, hei in neigh:
                points = face.getVertexesForDelTest(fi, hei)
                #isDel = checkDelaunay2(points[0], points[1])
                isDel = common.checkDelaunayFromCachedParams(face.circleParams, points[1])
                if not isDel:
                    self.flip(hei)
                         
    def triangulate(self, initFacesIds):
        #self.triangulateAll()
        #self.triangulateAll()
        #return
        #img = cv2.flip(self.draw(), 0)
        stack = list(initFacesIds)
        walked = [False for _ in self.faces]
        #for face in self.faces:
        while len(stack) > 0:
            faceIdx = stack.pop()
            if walked[faceIdx]:
                continue
            walked[faceIdx] = True
            face = self.faces[faceIdx]
            #if q:break
            neigh = face.getNeighFaces()
            isBadFace = False
            for fi, hei in neigh:
                #if hei in [1102, 1103]:
                #    print '>', fi, hei
                points = face.getVertexesForDelTest(fi, hei)
                #isDel = delcommon.checkDelaunay1(points[0], points[1])
                #isDel = checkDelaunay2(points[0], points[1])
                isDel = common.checkDelaunayFromCachedParams(face.circleParams, points[1])
                if not isDel:
                    #if hei in [1102, 1103]:
                    #    print 'bad', fi, hei
                    #print 'bad del1'
                    #points = self.hedges[hei].vertexesPoints()
                    #draw_arrow(img, points[1], points[0], (0, 255, 0))
                    #cv2.imshow('90', cv2.flip(img, 0))
                    #print fi, hei
                    #print mesh.hedges
                    #cv2.waitKey()

                    isBadFace = True
                    self.flip(hei)
                    #stack.append(fi)
                    #walked[fi] = False
                    for fi2, _ in self.faces[fi].getNeighFaces():
                        stack.append(fi2)
                    #q = True
                    #break
                    #commonHEdge = self.hedges[hei]
                    #commonHEdgeOp = commonHEdge.opposite()
                    #face2 = self.faces[fi]
                    #points = face2.getVertexesForDelTest(face.index, commonHEdgeOp.index)
                    #isDel2 = checkDelaunay2(points[0], points[1])
                    #if not isDel2:
                    #    #print 'flipping'
                    #    self.flip(hei)
                    #'''
            if isBadFace:
                #stack.append(face.index)
                #walked[face.index] = False
                for fi2, _ in face.getNeighFaces():
                    stack.append(fi2)
            #if isBadFace:
        #cv2.imshow('90', cv2.flip(img, 0))
        self.triangulateAll()
        #self.triangulateAll()

    def addPointAndTriangulate(self, point):
        #print 'Searching'
        triIndx = self.findTriangleByPoint(point)
        if triIndx == None:
            print 'bad point', point
            return
        
        fcs = self.addPointInsideTriangle(triIndx, point)
        #print 'Triangulating'
        self.triangulate(fcs)
        #self.triangulate()

    def __repr__(self):
        return 'Mesh (faces (%s)\nVertexes (%s)\nHEdges (%s) \n)' % (
                    '\n'.join(map(str, self.faces)),
                    '\n'.join(map(str, self.vertexes)),
                    '\n'.join(map(str, self.hedges))
                )

