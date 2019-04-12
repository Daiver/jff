import unittest

import geometry_tools


class ObjTests(unittest.TestCase):
    def test_triangulate01(self):
        faces = [
            [5, 3, 5]
        ]
        res = geometry_tools.tools.triangulate_polygons(faces)
        ans = [[5, 3, 5]]
        self.assertEquals(res, ans)

    def test_triangulate02(self):
        faces = [
            [5, 3, 6, 5]
        ]
        res = geometry_tools.tools.triangulate_polygons(faces)
        ans = [
            [5, 3, 6],
            [5, 6, 5],
        ]
        self.assertEquals(res, ans)

    def test_triangulate03(self):
        faces = [
            [11, 5, 3, 6, 5],
            [0, 3, 4],
            [1, 2, 3, 4]
        ]
        res = geometry_tools.tools.triangulate_polygons(faces)
        ans = [
            [11, 5, 3],
            [11, 3, 6],
            [11, 6, 5],
            [0, 3, 4],
            [1, 2, 3],
            [1, 3, 4]
        ]
        self.assertEquals(res, ans)


if __name__ == '__main__':
    unittest.main()
