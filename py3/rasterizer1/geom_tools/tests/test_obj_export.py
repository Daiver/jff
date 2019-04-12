import unittest

import io
import numpy as np
import geom_tools


class TestObjParser(unittest.TestCase):
    def test_mesh_obj_export01(self):
        mesh = geom_tools.Mesh(
            vertices=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=np.float32),
            polygon_vertex_indices=[
                [0, 1, 2]
            ],
        )

        stream = io.StringIO()
        geom_tools.save_to_stream(stream, mesh)
        res = stream.getvalue()
        ans = """\
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0

f 1 2 3
"""
        self.assertEqual(res, ans)

    def test_mesh_obj_export02(self):
        mesh = geom_tools.Mesh(
            vertices=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [5, 6, 7.5],
            ]),
            polygon_vertex_indices=[
                [0, 1, 2, 3],
                [3, 1, 2],
            ],
        )

        stream = io.StringIO()
        geom_tools.save_to_stream(stream, mesh)
        res = stream.getvalue()
        ans = """\
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
v 5.0 6.0 7.5

f 1 2 3 4
f 4 2 3
"""
        self.assertEqual(res, ans)


if __name__ == '__main__':
    unittest.main()
