import unittest

import numpy as np
from geom_tools import Mesh


class MeshTests(unittest.TestCase):
    def test_mesh_comparison01(self):
        mesh1 = Mesh(vertices=np.arange(3), polygon_vertex_indices=[0, 1, 2])
        mesh2 = Mesh(vertices=np.arange(3), polygon_vertex_indices=[0, 1, 2])
        self.assertTrue(mesh1 == mesh2)

