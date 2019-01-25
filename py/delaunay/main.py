import cv2
import numpy as np
import math

class Vertex:
    def __init__(self, point, neigh=None):
        self.point = point
        self.neigh = neigh if neigh else []

    def __repr__(self):
        return 'Vertex((%d, %d) %s)' % (self.point[0], self.point[1], str(self.neigh))

def vizVertexes(vertexs):
    mxX = max(vertexs, key=lambda x:x.point[0]).point[0]
    mxY = max(vertexs, key=lambda x:x.point[1]).point[1]
    border = 30
    img = np.ones((mxY + border, mxX + border, 3))*255
    for v in vertexs:
        for v2ind in v.neigh:
            v2 = vertexs[v2ind]
            if v != v2:
                cv2.line(img, v.point, v2.point, (255, 0, 0), 2)
    for v in vertexs:
        cv2.circle(img, v.point, 5, (0, 255, 0), 3)
    return img

def divideListByHalf(lst):
    if len(lst) < 2:
        return lst, []
    thr = int(math.ceil(len(lst)/2.0))
    return lst[:thr], lst[thr:]


def delanaunay(vertexs):
    def divideAndMerge(vertexsIndxs):
        print vertexsIndxs
        a, b = divideListByHalf(vertexsIndxs)
        newA = divideAndMerge(a) if(len(a) > 3) else baseComb(a)
        newB = divideAndMerge(b) if(len(b) > 3) else baseComb(b)
        if len(newA) == 0 or len(newB) == 0: return max(newA, newB)

        #merge
        newA.sort(key=lambda i: vertexs[i].point[1])
        newB.sort(key=lambda i: vertexs[i].point[1])

        #make baseline
        vertexs[newA[-1]].neigh.append(newB[-1])
        vertexs[newB[-1]].neigh.append(newA[-1])
        return newA + newB

    def baseComb(lst):
        if len(lst) > 1:
            vertexs[lst[0]].neigh.append(lst[1])
            vertexs[lst[1]].neigh.append(lst[0])
            if len(lst) == 3:
                vertexs[lst[0]].neigh.append(lst[2])
                vertexs[lst[2]].neigh.append(lst[0])
                vertexs[lst[1]].neigh.append(lst[2])
                vertexs[lst[2]].neigh.append(lst[1])
        return lst
        
    vertexs.sort(key=lambda x: (x.point[0], -x.point[1]))
    divideAndMerge(range(len(vertexs)))

if __name__ == '__main__':
    vertexs = [
                Vertex((10, 80)),
                Vertex((60, 140)),
                Vertex((100, 80)),
                Vertex((190, 140)),
                Vertex((190, 80)),
                Vertex((160, 50)),
                Vertex((190, 10)),
                Vertex((130, 10)),
                Vertex((60, 50)),
                Vertex((60, 10)),
            ]
    delanaunay(vertexs)
    cv2.imshow('', vizVertexes(vertexs))
    print vertexs
    cv2.waitKey()

