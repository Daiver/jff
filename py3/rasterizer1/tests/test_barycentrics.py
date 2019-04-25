import unittest

import numpy as np
import rasterizer_cpp


class TestBarycentrics(unittest.TestCase):
    def test_barycentrics_from_2d_vertices_01(self):
        res = np.array(rasterizer_cpp.barycoords_from_2d_trianglef(
            0, 0,
            1, 0,
            0, 1,
            0.5, 0.5), dtype=np.float32)
        self.assertTrue(np.allclose(res, [0, 0.5, 0.5]))

    def test_barycentrics_from_2d_vertices_02(self):
        res = np.array(rasterizer_cpp.barycoords_from_2d_trianglef(
            0, 0,
            1, 0,
            0, 1,
            0, 0), dtype=np.float32)
        self.assertTrue(np.allclose(res, [1, 0, 0]))

    def test_barycentrics_from_2d_vertices_03(self):
        res = np.array(rasterizer_cpp.barycoords_from_2d_trianglef(
            0, 0,
            1, 0,
            0, 1,
            1, 0), dtype=np.float32)
        self.assertTrue(np.allclose(res, [0, 1, 0]))

    def test_barycentrics_from_2d_vertices_04(self):
        res = np.array(rasterizer_cpp.barycoords_from_2d_trianglef(
            0, 0,
            1, 0,
            0, 1,
            0, 1), dtype=np.float32)
        self.assertTrue(np.allclose(res, [0, 0, 1]))

    def test_barycentrics_from_2d_vertices_05(self):
        res = np.array(rasterizer_cpp.barycoords_from_2d_trianglef(
            0, 0,
            1, 0,
            0, 1,
            0.5, 0), dtype=np.float32)
        self.assertTrue(np.allclose(res, [0.5, 0.5, 0]))

    def test_barycentrics_from_2d_vertices_06(self):
        res = np.array(rasterizer_cpp.barycoords_from_2d_trianglef(
            0, 0,
            1, 0,
            0, 1,
            0, 0.5), dtype=np.float32)
        self.assertTrue(np.allclose(res, [0.5, 0, 0.5]))


if __name__ == '__main__':
    unittest.main()
