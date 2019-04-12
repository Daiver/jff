import unittest

import numpy as np
import geom_tools


class TestObjParser(unittest.TestCase):
    def test_from_obj_string01(self):
        content = """
        v 1 0 0
        v 0 1 0
        v 0 0 1
        
        f 1 2 3
        
        """
        res = geom_tools.from_obj_string(content)
        ans = geom_tools.Mesh(
            vertices=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            polygon_vertex_indices=[[0, 1, 2]],
            triangle_vertex_indices=[[0, 1, 2]]
        )
        self.assertTrue(res == ans)


if __name__ == '__main__':
    unittest.main()
