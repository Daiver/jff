import unittest

import numpy as np
import geom_tools
import torch

from torch_rasterizer import mk_rasterizer


class TestTorchRasterizer(unittest.TestCase):
    def test_forward01(self):
        canvas_size = (5, 5)
        model = geom_tools.Mesh(
            vertices=np.array([
                [0, 0, 1],
                [4, 0, 1],
                [0, 4, 0],
            ], dtype=np.float32),
            polygon_vertex_indices=[
                [0, 1, 2]
            ],
            texture_vertices=np.array([
                [0, 1],
                [1, 0],
                [0, 1]
            ], dtype=np.float32),
            texture_polygon_vertex_indices=[
                [0, 1, 2]
            ],
            triangle_vertex_indices=[
                [0, 1, 2]
            ],
            triangle_texture_vertex_indices=[
                [0, 1, 2]
            ],
        )

        rasterizer = mk_rasterizer(
            model.triangle_vertex_indices,
            model.triangle_texture_vertex_indices,
            canvas_size)
        vertices = torch.FloatTensor(model.vertices)
        texture = torch.zeros(canvas_size)
        _, z_buffer, bary, tri_indices = rasterizer(vertices, texture)

        print(z_buffer)
        print(bary.transpose(0, 2))
        print(tri_indices)

        ans_tri_indices = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1],
            [0, 0, 0, -1, -1],
            [0, 0, -1, -1, -1],
            [0, -1, -1, -1, -1],
        ]).int()
        self.assertTrue((tri_indices == ans_tri_indices).all())


if __name__ == '__main__':
    unittest.main()
