import numpy as np
import cv2
import sys
from deformer import *

def mkPoint(p, translate, scale):
    return (int(p[0] * scale + translate[0]), int(p[1] * scale + translate[1]))

def draw(verticesAdjacency, positions, translate, scale, img):
    nVertices = len(verticesAdjacency)
    for vInd in xrange(nVertices):
        for vInd2 in verticesAdjacency[vInd]:
            p1 = mkPoint(positions[vInd ], translate, scale)
            p2 = mkPoint(positions[vInd2], translate, scale)
            cv2.line(img, p1, p2, (0, 0, 255), 2)
    for p in positions:
        pPrime = mkPoint(p, translate, scale)
        cv2.circle(img, pPrime, 5, (0, 255, 0), 3)

def main1():
    ks = 1.0
    kb = 1.0
    adj = [
            [1, 5],
            [0, 2, 5],
            [1, 3, 4],
            [2, 4],
            [2, 3, 5],
            [0, 1, 4]
        ]

    positions = np.array([
        [0, 0],
        [1, 1],
        [2, 1],
        [3, 0],
        [2, -1],
        [1, -1]
        ], dtype=np.float)

    hardConstraintsIndices = [1, 5, 3]
    hardConstraintsDisplacementsX = [-2, -2, -3]
    L1 = composeL1Matrix(adj)
    L2 = composeL2Matrix(adj, L1)
    A0 = ks * L1 - kb * L2
    A = composeAMatFinal(A0, hardConstraintsIndices)
    B = composeRHS(A0, hardConstraintsIndices, hardConstraintsDisplacementsX)
    resX = np.linalg.solve(A, B)
    print L1
    print L2
    print A0
    print A
    print B
    print resX
    displacementsX = composeFinalDisplacements1D(hardConstraintsIndices, hardConstraintsDisplacementsX, resX)
    print 'displacementsX'
    print displacementsX

    canvas = np.zeros((500, 500, 3))
    draw(adj, positions, [150, 100], 50, canvas)
    cv2.imshow('', canvas)

    positions[:, 0] += displacementsX
    canvas = np.zeros((500, 500, 3))
    draw(adj, positions, [150, 100], 50, canvas)
    cv2.imshow('1', canvas)
    cv2.waitKey()

if __name__ == '__main__':
    main1()
