import unittest

import autograd.numpy as np
import contourloss

class TestContourLoss(unittest.TestCase):

    def testRemapPointsToSegments01(self):
        points  = np.array([[1, 2], [3, 4], [5, 6]])
        indices = np.array([[0, 1], [2, 0]])

        res     = contourloss.remapPointsToSegements(points, indices)
        ans     = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [1, 2]]
        ])
        self.assertEqual(res.shape, ans.shape)
        self.assertTrue(np.allclose(res, ans))

    def testPoints2LineLoss01(self):
        eps = 1e-5
        segments = np.array([
            [[0, 0], [1, 0]],
            [[1, 0], [1, -1]]
        ], dtype=np.float32)

        pointsToBring = np.array([
            [0.5, 10],
        ], dtype=np.float32)
        indices = [0]

        weights = np.ones(len(pointsToBring))
        res = contourloss.pointsForLineLoss(segments, pointsToBring, indices, weights)

        self.assertTrue(abs(res - 100.0) < eps)

    def testPoints2LineLoss02(self):
        eps = 1e-5
        segments = np.array([
            [[0, 0], [1, 0]],
            [[1, 0], [1, -1]]
        ], dtype=np.float32)

        pointsToBring = np.array([
            [2, 1],
        ], dtype=np.float32)
        indices = [1]

        weights = np.ones(len(pointsToBring))
        res = contourloss.pointsForLineLoss(segments, pointsToBring, indices, weights)
        self.assertTrue(abs(res - 1.0) < eps)

    def testPoints2LineLoss03(self):
        eps = 1e-5
        segments = np.array([
            [[0, 0], [1, 0]],
            [[1, 0], [1, -1]]
        ], dtype=np.float32)

        pointsToBring = np.array([
            [0.5, 10],
            [2, 1],
            [3, -0.5]
        ], dtype=np.float32)
        indices = [0, 1, 1]

        weights = np.ones(len(pointsToBring))
        res = contourloss.pointsForLineLoss(segments, pointsToBring, indices, weights)
        self.assertTrue(abs(res - (100.0 + 1 + 4)) < eps)

    def testSegmentsEqualityAll2AllLoss01(self):
        eps = 1e-5
        segments = np.array([
            [[0, 0], [1, 0]],
            [[1, 0], [1, -1]]
        ], dtype=np.float32)
        res = contourloss.segmentsLengthEquallityAll2AllLoss(segments)
        self.assertTrue(abs(res - (0.0)) < eps)

    def testSegmentsEqualityAll2AllLoss02(self):
        eps = 1e-5
        segments = np.array([
            [[0, 0], [1, 0]],
            [[0, 0], [5, 0]]
        ], dtype=np.float32)
        res = contourloss.segmentsLengthEquallityAll2AllLoss(segments)
        self.assertTrue(abs(res - (2*576.0)) < eps)

    def testSegmentsEqualityAll2AllLoss03(self):
        eps = 1e-5
        segments = np.array([
            [[0, 0], [1, 0]],
            [[5, 0], [0, 0]],
        ], dtype=np.float32)
        res = contourloss.segmentsLengthEquallityAll2AllLoss(segments)
        self.assertTrue(abs(res - (2*576.0)) < eps)

    def testSegmentsEqualityAll2AllLoss04(self):
        eps = 1e-5
        segments = np.array([
            [[1, 2], [3, 2]],
            [[5, 6], [5, 8]],
            [[10, 1], [10, -1]],
        ], dtype=np.float32)
        res = contourloss.segmentsLengthEquallityAll2AllLoss(segments)
        self.assertTrue(abs(res - (0)) < eps)

    def testSegmentsEqualityAll2AllLoss05(self):
        eps = 1e-5
        segments = np.array([
            [[0, 0], [0, 1]],
            [[0, 0], [0, 2]],
            [[0, 0], [0, 3]],
        ], dtype=np.float32)
        res = contourloss.segmentsLengthEquallityAll2AllLoss(segments)
        self.assertTrue(abs(res - 2*(64 + 25 + 9)) < eps)

if __name__ == '__main__':
    unittest.main()

