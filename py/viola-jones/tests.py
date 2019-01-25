import unittest

import cv2
import numpy as np

import haar
import find_best_thr
import dec_stump

class TestStringMethods(unittest.TestCase):
    def test_fbt01(self):
        weights = np.ones(6)
        values  = np.array([3, 2, 1, 4, 5, 0])
        labels  = np.array([0, 0, 0, 1, 1, 0])
        val, err, pol = find_best_thr.findBestThr(values, weights, labels)
        self.assertTrue(abs(err) < 0.00001)
        self.assertTrue(abs(val - 3.5) < 0.00001)
        self.assertTrue(abs(pol - 1) < 0.00001)

    def test_fbt02(self):
        weights = np.ones(6)
        values  = np.array([3, 2, 7, 4, 1, 0])
        labels  = np.array([0, 1, 0, 0, 1, 1])
        val, err, pol = find_best_thr.findBestThr(values, weights, labels)
        #print val, err, pol
        self.assertTrue(abs(err) < 0.00001)
        self.assertTrue(abs(val - 2.5) < 0.00001)
        self.assertTrue(abs(pol + 1) < 0.00001)

    def test_stump01(self):
        a = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            ], dtype=np.float32)
        aInt = cv2.integral(a)
        rect = [0, 0, 4, 4]
        negR, posR = haar.haarHorizLine(rect[0], rect[1], rect[2], rect[3])
        val = haar.computeHaarFeature(aInt, negR, posR)
        self.assertTrue(abs(val - 8.0) < 0.00001)

        stump = dec_stump.Stump('hor', rect, 6, 1)
        ans = stump.predict(aInt)
        self.assertEquals(ans, 1)

        stump = dec_stump.Stump('hor', rect, 9, 1)
        ans = stump.predict(aInt)
        self.assertEquals(ans, 0)

        stump = dec_stump.Stump('hor', rect, 6, -1)
        ans = stump.predict(aInt)
        self.assertEquals(ans, 0)


unittest.main()
