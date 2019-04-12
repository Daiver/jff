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
            ]),
            polygon_vertex_indices=[
                [0, 1, 2]
            ],
        )

        stream = io.StringIO()
        geom_tools.save_to_stream(stream, mesh)
        res = stream.getvalue()
        ans = """\
v 1 0 0
v 0 1 0
v 0 0 1

f 1 2 3
"""
        self.assertEqual(res, ans)


if __name__ == '__main__':
    unittest.main()
