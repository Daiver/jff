
import sys

def newSegmentPolygons(pInd1, pInd2, lastInd):
    nInd1 = lastInd + 1
    nInd2 = lastInd + 2
    nInd3 = lastInd + 3
    return ([
            [pInd1, nInd3, nInd1],
            [nInd1, nInd3, nInd2],
            [pInd2, nInd2, nInd3],
            [pInd2, nInd3, pInd1]
        ], (nInd1, nInd2, nInd3))

def staticPositions(startOffset, step, nVertices):
    assert ((nVertices - 2) % 3 == 0)
    res = [
            [startOffset, 0.0, -1.0],
            [startOffset, 0.0,  1.0]
            ]

    nSegments = (nVertices - 2) / 3
    for i in xrange(0, nSegments):
        res += [
                [startOffset + (i + 1) * step,     0.0, -1.0],
                [startOffset + (i + 1) * step,     0.0,  1.0],
                [startOffset + (i + 0) * step + step/2.0, 0.0,  0.0]
                ]
    return res

def genObj(startOffset, step, nSegments):
    nVertices = 2 + nSegments * 3
    vertices = staticPositions(startOffset, step, nVertices)
    polygons = []
    pInd1 = 0
    pInd2 = 1
    lastInd = 1
    for i in xrange(nSegments):
        newSegments, newIndices = newSegmentPolygons(pInd1, pInd2, lastInd)
        polygons += newSegments
        pInd1, pInd2, lastInd = newIndices
    return vertices, polygons

def saveObj(fname, vertices, polygons):
    with open(fname, 'w') as f:
        for v in vertices:
            f.write('v %f %f %f \n' % (v[0], v[1], v[2]))
        f.write('\n')
        for p in polygons:
            string = 'f '
            for i in p:
                string += str(i + 1) + ' '
            string += '\n'
            f.write(string)

def addDisturbance(vertices, offset, step):
    nSegments = (len(vertices) - 2) / 3
    for i in xrange(nSegments):
        vertices[4 + 3 * i][1] = offset + i * step

if __name__ == '__main__':
    #print newSegmentPolygons(0, 1, 1) + newSegmentPolygons(2, 3, 4)
    #print staticPositions(0.0, 2.0, 5)
    #v, p = genObj(0.0, 2.0, 10000)
    #addDisturbance(v, 0.3, 0.01)
    v, p = genObj(0.0, 10.0, 10000)
    addDisturbance(v, 1.5, 0.01)
    saveObj('auto_slavik4.obj', v, p)

