import unittest
import numpy as np

from arap_image_warp1 import findCommonVertices, cellIndices, edgeLengths
from arap_image_warp1 import gMatrix, gMatrices, hMatrix
from arap_image_warp1 import composeA1Matrix, composeB1Matrix
from arap_image_warp1 import composeA2Matrix, composeB2Matrix
from arap_image_warp1 import normalizedTransformationFromPositions

class ARAPImageWarpTests01(unittest.TestCase):
    def setUp(self):
        self.adj = [
                [1, 3, 4],
                [0, 2, 4],
                [3, 4, 1],
                [2, 4, 0],
                [0, 1, 2, 3]
                ]
        self.pos = np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                [0.5, 0.5]
                ])

    def testComposeA2Matrix01(self):
        cells = [
                    [0, 1, 2],#0
                    [1, 2, 0],#1
                    [2, 0, 1],#2
                    [1, 0, 2],#3
                    [2, 1, 0],#4
                    [0, 2, 1] #5
                ]
        constraints = [0]
        nVerts = 3
        weight = 100
        res = composeA2Matrix(cells, constraints, nVerts, weight)
        ans = np.array([
            [-1, 1, 0],
            [0, -1, 1],
            [1, 0, -1],
            [1, -1, 0],
            [0, 1, -1],
            [-1, 0, 1],
            [100, 0, 0]
            ])
        self.assertTrue(np.allclose(res, ans))

    def testComposeB2Matrix01(self):
        weight = 100
        edgeLens1D = [1, 2, 3]
        controlPointDesirePositions = [1, 2]
        res = composeB2Matrix(edgeLens1D, controlPointDesirePositions, weight)
        ans = np.array([
            1, 2, 3, 100, 200
            ])
        self.assertTrue(np.allclose(res, ans))

    def testComposeA2Matrix02(self):
        cells = [
                    [0, 1, 2],#0
                    [1, 2, 0],#1
                    [2, 0, 1],#2
                    [1, 0, 2],#3
                    [2, 1, 0],#4
                    [0, 2, 1] #5
                ]
        constraints = [0]
        nVerts = 3
        weight = 100
        A2 = composeA2Matrix(cells, constraints, nVerts, weight)   

    def testNormalizedTransformationFromPositions01(self):
        pos = np.array([
            [0, 0],
            [0, 1],
            [-1, 0],
            [1, 0]
            ], dtype=np.float32)
        cell = [0, 1, 2, 3]
        newPos = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [0, -1]
            ], dtype=np.float32)
        g = gMatrix(pos, cell)
        trans = normalizedTransformationFromPositions(newPos, g, cell)
        for p, newp in zip(pos, newPos):
            self.assertTrue(np.allclose( np.dot(trans, p), newp))

    def testNormalizedTransformationFromPositions02(self):
        pos = np.array([
            [0, 0],
            [0, 1],
            [-1, 0],
            [1, 0]
            ], dtype=np.float32)
        cell = [0, 1, 2, 3]
        newPos = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [0, -1]
            ], dtype=np.float32)
        g = gMatrix(pos, cell)
        trans = normalizedTransformationFromPositions(newPos, g, cell)
        '''print np.dot(trans, pos[1])
        print np.dot(trans, pos[0])
        print np.dot(trans, pos[1] - pos[0]) '''

        self.assertTrue(np.allclose(np.dot(trans, pos[1]) - np.dot(trans, pos[0]), 
            np.dot(trans, pos[1] - pos[0])))

    def testfindCommonVertices01(self):
        res1 = findCommonVertices(self.adj, 0, 1)
        self.assertEqual(len(res1), 1)
        self.assertEqual(res1[0], 4)

        res2 = findCommonVertices(self.adj, 2, 4)
        self.assertEqual(len(res2), 2)
        self.assertEqual(res2[0], 3)
        self.assertEqual(res2[1], 1)

        res3 = findCommonVertices(self.adj, 4, 2)
        self.assertEqual(len(res3), 2)
        self.assertEqual(res3[0], 1)
        self.assertEqual(res3[1], 3)

    def testCellIndices01(self):
        res  = cellIndices(self.adj)
        true = [[0, 1, 4], [0, 3, 4], [0, 4, 1, 3], 
                [1, 0, 4], [1, 2, 4], [1, 4, 0, 2],
                [2, 3, 4], [2, 4, 3, 1], [2, 1, 4], 
                [3, 2, 4], [3, 4, 2, 0], [3, 0, 4], 
                [4, 0, 1, 3], [4, 1, 0, 2], [4, 2, 1, 3], [4, 3, 0, 2]]
        for x, y in zip(res, true):
            self.assertSequenceEqual(x, y)

    def testEdgeLength01(self):
        res = edgeLengths(self.pos, self.adj)
        ans = np.array([
                [1, 0],
                [0, 1],
                [0.5, 0.5],
                [-1, 0],
                [0, 1],
                [-0.5, 0.5],
                [-1, 0],
                [-0.5, -0.5],
                [0, -1],
                [1, 0],
                [0.5, -0.5],
                [0, -1],
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
                ])
        for x, y in zip(res, ans):
            self.assertEqual(x[0], y[0])
            self.assertEqual(x[1], y[1])

    def testGMatrix01(self):
        cell = [0, 1, 2]
        pos  = np.array([
                [0, 10], 
                [20, 30], 
                [40, 50]])
        res  = gMatrix(pos, cell)
        ans  = np.array([
                [0, 10],
                [10, 0],
                [20, 30],
                [30, -20],
                [40, 50],
                [50, -40],
            ])
        for x, y in zip(res, ans):
            self.assertEqual(x[0], y[0])

    def testHMatrix01(self):
        #return #test later
        pos  = np.array([
            [1, 2],
            [3, 2],
            [1.5, 1],
            [1.5, 3]
            ])
        e    = np.array([2, 0])
        cell = [0, 1, 2, 3]
        g    = gMatrix(pos, cell)
        res  = hMatrix(e, len(cell), g)
        #print 'G'
        #print g
        #print 'H'
        #print res

    def testComposeA1(self):
        h      = [
                np.array([
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]
                    ]),
                np.array([
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2]
                    ]),
                np.array([
                    [3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3, 3, 3, 3]
                    ])
                ]
        cells  = [
                [6, 3, 4, 2],
                [5, 4, 6],
                [0, 1, 6]
                ]
        nVerts = 7
        weight = 13
        conds  = [6, 2]
        res    = composeA1Matrix(h, cells, nVerts, weight, conds)
        ans = np.array([
            [[  0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   0,   0,   1,   1],
             [  0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   0,   0,   1,   1],
             [  0,   0,   0,   0,   0,   0,   0,   0,   2,   2,   2,   2,   2,   2],
             [  0,   0,   0,   0,   0,   0,   0,   0,   2,   2,   2,   2,   2,   2],
             [  3,   3,   3,   3,   0,   0,   0,   0,   0,   0,   0,   0,   3,   3],
             [  3,   3,   3,   3,   0,   0,   0,   0,   0,   0,   0,   0,   3,   3],
             [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  13,   0],
             [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  13],
             [  0,   0,   0,   0,  13,   0,   0,   0,   0,   0,   0,   0,   0,   0],
             [  0,   0,   0,   0,   0,  13,   0,   0,   0,   0,   0,   0,   0,   0]]
            ])

        self.assertTrue(np.allclose(ans, res))

    def testComposeB1(self):
        nCells      = 3
        weight      = 13
        constraints = np.array([
            [1, 2],
            [-2, 3]
            ])

        res = composeB1Matrix(nCells, weight, constraints)

        ans = np.array([
            0, 0, 0, 0, 0, 0, 
            13*1, 13*2, -2*13, 3*13
            ]).transpose()
        self.assertTrue(np.allclose(res, ans))

if __name__ == '__main__':
    unittest.main()
