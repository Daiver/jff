import numpy as np
import common

class Face:
    def __init__(self, vertexes, faces, hedges, _edge):
        self.vertexes = vertexes
        self.faces = faces
        self.hedges = hedges
        self._edge = _edge
        self.circleParams = None

    def edge(self): return self.hedges[self._edge]
    
    def allHEdges(self):
        st = self.edge()
        res = [st]
        cur = st.next()
        while cur != st:
            res.append(cur)
            cur = cur.next()
        return res

    def calculateCircleParams(self):
        points = self.vertexesPoints()
        self.circleParams = common.computeCircleParams(points)
        return self.circleParams

    def vertexesPoints(self):
        return map(lambda x: x.head().point, self.allHEdges())

    def __repr__(self):
        return 'Face %d(%s)' % (self.index, str(self._edge))

    def getVertexesForDelTest(self, testFace, commonHEdgeInd):
        res = []
        commonHEdge = self.hedges[commonHEdgeInd]
        p0 = commonHEdge.opposite().next().head().point
        p3 = commonHEdge.head().point
        p2 = commonHEdge.next().head().point
        p1 = commonHEdge.next().next().head().point
        return (p1, p2, p3), p0

    def getNeighFaces(self):
        res = []
        headHEdge = self.edge()
        op = headHEdge.opposite()._leftFace
        #print headHEdge, op
        if op != None:
            res.append((op, headHEdge.index))
        curHEdge = headHEdge.next()
        while curHEdge != headHEdge:
            op = curHEdge.opposite()._leftFace
            if op != None: res.append((op, curHEdge.index))
            curHEdge = curHEdge.next()
        return res

    def isPointBelong(self, p0):
        points = self.vertexesPoints()
        exp1 = ((points[0][0] - p0[0])*(points[1][1] - points[0][1]) 
             - (points[1][0] - points[0][0])*(points[0][1] - p0[1]))

        exp2 = ((points[1][0] - p0[0])*(points[2][1] - points[1][1]) 
             - (points[2][0] - points[1][0])*(points[1][1] - p0[1]))

        exp3 = ((points[2][0] - p0[0])*(points[0][1] - points[2][1]) 
             - (points[0][0] - points[2][0])*(points[2][1] - p0[1]))

        #print exp1, exp2, exp3
        return np.sign(exp1) == np.sign(exp2) and np.sign(exp2) == np.sign(exp3)


