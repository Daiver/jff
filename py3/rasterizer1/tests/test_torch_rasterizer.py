import unittest

import numpy as np
import torch
import geom_tools

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
            texture_vertices=np.array([
                [0, 1],
                [1, 0],
                [0, 1]
            ], dtype=np.float32),
            polygon_vertex_indices=[
                [2, 1, 0]
            ],
            texture_polygon_vertex_indices=[
                [2, 1, 0]
            ],
            triangle_vertex_indices=[
                [2, 1, 0]
            ],
            triangle_texture_vertex_indices=[
                [2, 1, 0]
            ],
        )

        rasterizer = mk_rasterizer(
            model.triangle_vertex_indices,
            model.triangle_texture_vertex_indices,
            torch.FloatTensor(model.texture_vertices),
            canvas_size,
            return_z_buffer=True,
            return_barycentrics=True,
        )
        vertices = torch.FloatTensor(model.vertices)
        texture = torch.zeros((canvas_size[0], canvas_size[1], 3))
        _, z_buffer, bary, tri_indices = rasterizer(vertices, texture)
        bary = bary.transpose(0, 2)

        ans_tri_indices = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1],
            [0, 0, 0, -1, -1],
            [0, 0, -1, -1, -1],
            [0, -1, -1, -1, -1],
        ]).int()
        ans_z_buffer = torch.FloatTensor([
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [0.7500, 0.7500, 0.7500, 0.7500, 0.0000],
            [0.5000, 0.5000, 0.5000, 0.0000, 0.0000],
            [0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
        ans_bary = torch.FloatTensor([
            [[0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
             [0.0000, 0.2500, 0.5000, 0.7500, 0.0000],
             [0.0000, 0.2500, 0.5000, 0.0000, 0.0000],
             [0.0000, 0.2500, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

            [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.2500, 0.2500, 0.2500, 0.2500, 0.0000],
             [0.5000, 0.5000, 0.5000, 0.0000, 0.0000],
             [0.7500, 0.7500, 0.0000, 0.0000, 0.0000],
             [1.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

            [[1.0000, 0.7500, 0.5000, 0.2500, -0.0000],
             [0.7500, 0.5000, 0.2500, 0.0000, 0.0000],
             [0.5000, 0.2500, 0.0000, 0.0000, 0.0000],
             [0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
        ])
        self.assertTrue((tri_indices == ans_tri_indices).all())
        self.assertTrue((ans_bary - bary).norm() < 1e-6)
        self.assertTrue((ans_z_buffer - z_buffer).norm() < 1e-6)

    def test_forward02(self):
        canvas_size = (5, 5)
        model = geom_tools.Mesh(
            vertices=np.array([
                [0, 0, 1],
                [4, 0, 0],
                [0, 4, 0],
                [4, 4, -1],
            ], dtype=np.float32),
            texture_vertices=np.array([
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 1],
            ], dtype=np.float32),
            polygon_vertex_indices=[
                [2, 1, 0],
                [2, 3, 1],
            ],
            texture_polygon_vertex_indices=[
                [2, 1, 0],
                [2, 3, 1],
            ],
            triangle_vertex_indices=[
                [2, 1, 0],
                [2, 3, 1],
            ],
            triangle_texture_vertex_indices=[
                [2, 1, 0],
                [2, 3, 1],
            ],
        )

        rasterizer = mk_rasterizer(
            model.triangle_vertex_indices,
            model.triangle_texture_vertex_indices,
            torch.FloatTensor(model.texture_vertices),
            canvas_size,
            return_z_buffer=True,
            return_barycentrics=True,
        )
        vertices = torch.FloatTensor(model.vertices)
        texture = torch.zeros((canvas_size[0], canvas_size[1], 3))
        _, z_buffer, _, tri_indices = rasterizer(vertices, texture)

        ans_tri_indices = torch.tensor([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]).int()
        ans_z_buffer = torch.FloatTensor([
            [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            [0.7500, 0.5000, 0.2500, 0.0000, -0.2500],
            [0.5000, 0.2500, 0.0000, -0.2500, -0.5000],
            [0.2500, 0.0000, -0.2500, -0.5000, -0.7500],
            [0.0000, -0.2500, -0.5000, -0.7500, -1.0000]])

        self.assertTrue((tri_indices == ans_tri_indices).all())
        self.assertTrue((ans_z_buffer - z_buffer).norm() < 1e-6)


if __name__ == '__main__':
    unittest.main()
