import unittest

import loss

import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad

class TestLoss(unittest.TestCase):
    def testDistPoint2Plane01(self):
        eps = 1e-5
        point = np.array([0, 1])
        p1    = np.array([0, 0])
        p2    = np.array([1, 0])
        res   = loss.distFromPoint2LineSq(point, p1, p2)
        self.assertTrue(abs(res - 1.0) < eps)

    def testDistPoint2Plane02(self):
        eps = 1e-5
        point = np.array([1, 2])
        p1    = np.array([0, 0])
        p2    = np.array([1, 0])
        res   = loss.distFromPoint2LineSq(point, p1, p2)
        self.assertTrue(abs(res - 4.0) < eps)

    def testDistPoint2Plane03(self):
        eps = 1e-5
        point = np.array([0, 1])
        p1    = np.array([0, 0])
        p2    = np.array([10000, 0])
        res   = loss.distFromPoint2LineSq(point, p1, p2)
        self.assertTrue(abs(res - 1.0) < eps)

    def testDistPoint2Plane04(self):
        eps = 1e-5
        point = np.array([0, 2])
        p1    = np.array([1, 10])
        p2    = np.array([1, 2])
        res   = loss.distFromPoint2LineSq(point, p1, p2)
        self.assertTrue(abs(res - 1.0) < eps)

    def testDistPoint2Plane05(self):
        eps = 1e-5
        point = np.array([-5, 2])
        p1    = np.array([1, 10])
        p2    = np.array([1, 2])
        res   = loss.distFromPoint2LineSq(point, p1, p2)
        self.assertTrue(abs(res - 36.0) < eps)

    def testDistPoint2Plane06(self):
        eps = 1e-5
        point = np.array([-5, 2])
        p1    = np.array([1, 10])
        p2    = np.array([1, 2])
        res   = loss.distFromPoint2LineSq(point, p1, p2)
        self.assertTrue(abs(res - 36.0) < eps)

    def testDistPoint2Plane07(self):
        eps = 1e-5
        point = np.array([0, 0])
        p1    = np.array([-1, -1])
        p2    = np.array([1, 1])
        res   = loss.distFromPoint2LineSq(point, p1, p2)
        self.assertTrue(abs(res - 0.0) < eps)

    def testDistPoint2Plane08(self):
        eps = 1e-5
        point = np.array([-1, 1])
        p1    = np.array([-1, -1])
        p2    = np.array([1, 1])
        res   = loss.distFromPoint2LineSq(point, p1, p2)
        self.assertTrue(abs(res - 2.0) < eps)

    def testDistPoint2Plane09(self):
        eps = 1e-5
        point = np.array([1, 1])
        p1    = np.array([-1, 1])
        p2    = np.array([1, -1])
        res   = loss.distFromPoint2LineSq(point, p1, p2)
        self.assertTrue(abs(res - 2.0) < eps)

    def testDistPoint2Plane10(self):
        eps = 1e-5
        point = np.array([-1, 1])
        p1    = np.array([-1, -1])
        p2    = np.array([1, 1])
        res   = loss.distFromPoint2LineSq(point, p1, p2)
        self.assertTrue(abs(res - 2.0) < eps)

    def testSegmentsLenEqualConstraint01(self):
        eps = 1e-5
        p1  = np.array([0, 0])
        p2  = np.array([2, 0])
        p3  = np.array([6, 5])
        p4  = np.array([6, 7])
        res = loss.segmentsEqualLengthConstraint(p1, p2, p3, p4)
        self.assertTrue(abs(res - 0.0) < eps)

    def testSegmentsLenEqualConstraint02(self):
        eps = 1e-5
        p1  = np.array([0, 0])
        p2  = np.array([2, 0])
        p3  = np.array([1, 1])
        p4  = np.array([2, 2])
        res = loss.segmentsEqualLengthConstraint(p1, p2, p3, p4)
        self.assertTrue(abs(res - 4.0) < eps)

    def testSegmentsLenEqualConstraint03(self):
        eps = 1e-5
        p1  = np.array([0, 0])
        p2  = np.array([2, 0])
        p3  = np.array([0, 0])
        p4  = np.array([0, 3])
        res = loss.segmentsEqualLengthConstraint(p1, p2, p3, p4)
        self.assertTrue(abs(res - 25.0) < eps)

    def testProjectPoint2Segment01(self):
        p1  = np.array([0, 0])
        p2  = np.array([1, 0])
        p   = np.array([0, 1])
        res = loss.projectPoint2Segment(p, p1, p2)
        ans = np.array([0, 0])
        self.assertTrue(np.allclose(res, ans))

    def testProjectPoint2Segment02(self):
        p1  = np.array([0, 0])
        p2  = np.array([1, 0])
        p   = np.array([1, 1])
        res = loss.projectPoint2Segment(p, p1, p2)
        ans = np.array([1, 0])
        self.assertTrue(np.allclose(res, ans))

    def testProjectPoint2Segment03(self):
        p1  = np.array([0, 0])
        p2  = np.array([1, 0])
        p   = np.array([1, -1])
        res = loss.projectPoint2Segment(p, p1, p2)
        ans = np.array([1, 0])
        self.assertTrue(np.allclose(res, ans))

    def testProjectPoint2Segment04(self):
        p1  = np.array([0, 0])
        p2  = np.array([1, 0])
        p   = np.array([0.2, -1])
        res = loss.projectPoint2Segment(p, p1, p2)
        ans = np.array([0.2, 0])
        self.assertTrue(np.allclose(res, ans))

    def testProjectPoint2Segment05(self):
        p1  = np.array([100, -7])
        p2  = np.array([100, 5])
        p   = np.array([100, -1000])
        res = loss.projectPoint2Segment(p, p1, p2)
        ans = np.array([100, -7])
        self.assertTrue(np.allclose(res, ans))

    def testProjectPoint2Segment06(self):
        p1  = np.array([100, -7])
        p2  = np.array([100, 5])
        p   = np.array([100, 1000])
        res = loss.projectPoint2Segment(p, p1, p2)
        ans = np.array([100, 5])
        self.assertTrue(np.allclose(res, ans))

    def testProjectPoint2Segment07(self):
        p1  = np.array([100, -7])
        p2  = np.array([100, 5])
        p   = np.array([1, 1000])
        res = loss.projectPoint2Segment(p, p1, p2)
        ans = np.array([100, 5])
        self.assertTrue(np.allclose(res, ans))

if __name__ == '__main__':
    unittest.main()

