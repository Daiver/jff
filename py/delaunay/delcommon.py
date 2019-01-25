import numpy as np

def checkDelaunay1(points, p):
    x0, y0 = p
    (x1, y1), (x2, y2), (x3, y3) = points
    exp = (((x0 - x1)*(y0 - y3) - (x0 - x3)*(y0 - y1)) *
            ((x2 - x1)*(x2 - x3) + (y2 - y1)*(y2 - y3)) +
            ((x0 - x1)*(x0 - x3) + (y0 - y1)*(y0 - y3)) *
            ((x2 - x1)*(y2 - y3) - (x2 - x3)*(y2 - y1))
           ) 
    #print 'exp', exp
    return exp >= 0

class Triangle:

    def __init__(self, allVertexes, allTriangles, vertexesIndxs, trianglesIndx=None):
        self.allVertexes = allVertexes
        self.allTriangles = allTriangles

        self.vertexesIndxs = vertexesIndxs
        self.trianglesIndx = trianglesIndx if trianglesIndx else []

    def __repr__(self):
        return 'Triangle(%s, %s)' % (str(self.vertexesIndxs), str(self.trianglesIndx))

    def getOppositeVertexInd(self, vertexesIndxs, enternalOpposite=None):
        vrtI = set(vertexesIndxs)
        if enternalOpposite != None and enternalOpposite in self.vertexesIndxs:
            return None
        numOfMatch = 0
        opposite = None
        for vi in self.vertexesIndxs:
            if vi in vrtI:
                numOfMatch += 1
            else:
                opposite = vi
        #if opposite = None: 
        #    return None
        #    raise Exception("no opposite")
        if numOfMatch != 2: 
            #raise Exception("bad numOfMatch %d" % numOfMatch)
            return None
        return opposite

    def checkDelaunay(self):
        #a = [a[-1]] + a[:-1]]
        vertexesIndxs = self.vertexesIndxs[:]
        triangles = map(lambda t: self.allTriangles[t], self.trianglesIndx)
        vertexes  = map(lambda v: self.allVertexes[v], vertexesIndxs)
        vertexes = [vertexes[-1]] + vertexes[:-1]
        for vi in self.vertexesIndxs:
            vertexes = [vertexes[-1]] + vertexes[:-1]
            print vertexes
            #print triangles[0].getOppositeVertexInd([0,1,2], 1)
            #print map(lambda x: x.getOppositeVertexInd(self.vertexesIndxs, vi), triangles)
            oppositeVInd = max(
                            map(lambda x: x.getOppositeVertexInd(self.vertexesIndxs, vi),
                            triangles))
            #if oppositeVInd == None: #potential dangeour
            #    continue
            print oppositeVInd, vi
            if ((oppositeVInd != None) and
                (not checkDelaunay1(vertexes, self.allVertexes[oppositeVInd]))):
                print 'returned'
                return False
            
        return True

if __name__ == '__main__':
    vertexes = [
                (0., 1.),
                (3.5, 0.),
                (6., 1.),
                (2.5, 10.),
            ]
    triangles = []
    triangles += [
                #Triangle(vertexes, triangles, [1, 0, 3], [1]),
                #Triangle(vertexes, triangles, [3, 2, 1], [0]),
                Triangle(vertexes, triangles, [2, 1, 0], [1]),
                Triangle(vertexes, triangles, [0, 3, 2], [0]),
            ]

    print triangles
    print triangles[1].getOppositeVertexInd([1,0,3], 0)
    print triangles[0].checkDelaunay()
    print triangles[1].checkDelaunay()
    #print checkDelaunay1(vertexes[:-1], (5, 0))
