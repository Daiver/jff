import unittest

import geom_tools


class ObjParserTests(unittest.TestCase):
    def test_triangulate01(self):
        faces = [
            [5, 3, 5]
        ]
        res = geom_tools.utils.triangulate_polygons(faces)
        ans = [[5, 3, 5]]
        self.assertTrue(res == ans)

    def test_triangulate02(self):
        faces = [
            [5, 3, 6, 5]
        ]
        res = geom_tools.utils.triangulate_polygons(faces)
        ans = [
            [5, 3, 6],
            [5, 6, 5],
        ]
        self.assertTrue(res == ans)

    def test_triangulate03(self):
        faces = [
            [11, 5, 3, 6, 5],
            [0, 3, 4],
            [1, 2, 3, 4]
        ]
        res = geom_tools.utils.triangulate_polygons(faces)
        ans = [
            [11, 5, 3],
            [11, 3, 6],
            [11, 6, 5],
            [0, 3, 4],
            [1, 2, 3],
            [1, 3, 4]
        ]
        self.assertTrue(res == ans)


if __name__ == '__main__':
    unittest.main()
