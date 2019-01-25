import unittest

import numpy as np
import random
import matplotlib.pyplot as plt

import fern
import dataprocessing
import CPR

class FernTests(unittest.TestCase):
    
    def testRanges(self):
        data = np.array([
            [1, 2, 3, 4],
            [-2, 10, 6, 5],
            [-1, 0, 15, 8]], dtype=np.float32)
        mins, maxs = fern.getRangesFromData(data)
        self.assertTrue(np.allclose(mins, np.array([-2, 0, 3, 4], dtype=np.float32)))
        self.assertTrue(np.allclose(maxs, np.array([ 1, 10, 15, 8], dtype=np.float32)))

#    def testPredictWithBorders01(self):
        #fr = fern.FernRegressor(3)
        #fr.bins = np.array([1., 2., 3., 4., 5., 6., 7., 8.], dtype=np.float32)
        #fr.thresholds = np.array([0.5, 1.0, -0.5])
        #self.assertEqual(fr.predict(np.array([1., 0.1, 2], dtype=np.float32))[0], 6.0)
        #self.assertEqual(fr.predict(np.array([1., 1.1, 2], dtype=np.float32))[0], 8.0)
        #self.assertEqual(fr.predict(np.array([0.1, 1.1, 2], dtype=np.float32))[0], 4.0)
        #self.assertEqual(fr.predict(np.array([0.1, 0.1, 2], dtype=np.float32))[0], 2.0)
        #self.assertEqual(fr.predict(np.array([0.1, 0.1, -2], dtype=np.float32))[0], 1.0)
        ##1 0 1 # 5

    def testRandomFernRegressor01(self):
        random.seed(42)
        reg = fern.RandomFernRegressor(2)
        data = np.array([
            [-1, 0, 5, -3, 6],
            [-1, 6, -5, -3, -2],
            [1, -1, -5, 3, 6]
            ], dtype=np.float32)
        values = np.array([5, 4, 6])
        #RFern([[ 4.50738907  5.          5.          5.97087383]] [[3, 0]])
        reg.fit(data, values)
        self.assertTrue(np.linalg.norm(
            reg.bins - np.array([ 4.50738907, 5., 5., 5.97087383])) < 0.01)

    #def testBoostedFern01(self):
        #reg = fern.FernRegressorBoosted(1, 50, 5)
        #f = lambda x: np.cos(x[0] * np.pi * 4) * (x[0] + 1)**2 
        #x1 = np.array([[float(i)] for i in xrange(10)])
        #reg.fit(x1, map(f, x1))

        #x2 = np.array([[i/10.0] for i in xrange(100)])
        #y  = map(f, x2)
        #r  = reg.predict(x2)
        #for f in reg.ferns:
            #print f

        #print map(f, x1)

    def testComputePoseFromTriangle01(self):
        p1 = np.array([-1., -1./3], dtype=np.float32)
        p2 = np.array([ 1., -1./3], dtype=np.float32)
        p3 = np.array([ 0.,  2./3], dtype=np.float32)
        pose = dataprocessing.computePoseFromTriangle(p1, p2, p3)
        self.assertTrue(np.linalg.norm(pose - np.array([0.0, 0.0, 0.0, 1.0, 1.0])) < 0.001)

    def testComputePoseFromTriangle02(self):
        p1 = np.array([-1., -1.], dtype=np.float32)
        p2 = np.array([ 1., -1.], dtype=np.float32)
        p3 = np.array([ 0.,  2.], dtype=np.float32)
        pose = dataprocessing.computePoseFromTriangle(p1, p2, p3)
        self.assertTrue(np.linalg.norm(pose - np.array([0.0, 0.0, 0.0, 1.0, 3.0])) < 0.001)
        point = dataprocessing.apply5DTransformation(pose, p1)
        self.assertTrue(np.linalg.norm(point - np.array([-1., -1./3])))
        
    def testComputePoseFromTriangle03(self):
        p1 = np.array([ 0.5, -1.], dtype=np.float32)
        p2 = np.array([ 1.5, -1.], dtype=np.float32)
        p3 = np.array([ 1.,  2.], dtype=np.float32)
        pose = dataprocessing.computePoseFromTriangle(p1, p2, p3)
        self.assertTrue(np.linalg.norm(pose - np.array([1.0, 0.0, 0.0, 0.5, 3.0])) < 0.001)
        point = dataprocessing.apply5DTransformation(pose, p1)
        self.assertTrue(np.linalg.norm(point - np.array([-1., -1./3])))

    def testComputePoseFromTriangle04(self):
        p1 = np.array([ 2.0, -2.0], dtype=np.float32)
        p2 = np.array([ 2.0,  2.0], dtype=np.float32)
        p3 = np.array([ 0.,  0.], dtype=np.float32)
        pose = dataprocessing.computePoseFromTriangle(p1, p2, p3)
        self.assertTrue(np.linalg.norm(pose - np.array([4./3, 0.0, 1.57079637, 2.0, 2.0])) < 0.001)
        point = dataprocessing.apply5DTransformation(pose, p1)
        self.assertTrue(np.linalg.norm(point - np.array([-1., -1./3])))

    def testNormalizeAngle01(self):
        angle = 3.0
        self.assertEqual(CPR.normalizeAngle(angle), 3.0)
        angle = -3.0
        self.assertEqual(CPR.normalizeAngle(angle), -3.0)
        angle = -4.0
        self.assertEqual(CPR.normalizeAngle(angle), np.pi*2 -4.0)

if __name__ == '__main__':
    unittest.main()

