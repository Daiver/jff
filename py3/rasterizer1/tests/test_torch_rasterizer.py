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
        texture = texture.permute(2, 0, 1)
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
        texture = texture.permute(2, 0, 1)
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

    def test_forward03(self):
        canvas_size = (5, 5)
        model = geom_tools.Mesh(
            vertices=np.array([
                [0, 0, 1],
                [4, 0, 0],
                [0, 4, 0],
                [4, 4, -1],
            ], dtype=np.float32),
            texture_vertices=np.array([
                [0, 0],
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
        texture = torch.FloatTensor([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]).view(3, 3, 1)
        texture = texture.permute(2, 0, 1)
        render, _, _, tri_indices = rasterizer(vertices, texture)

        ans_tri_indices = torch.tensor([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]).int()

        ans_render = torch.FloatTensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])

        self.assertTrue((tri_indices == ans_tri_indices).all())
        self.assertTrue((render - ans_render).norm() < 1e-6)

    def test_forward04(self):
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
                [1, 1],
                [0, 0],
                [1, 0],
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
            return_z_buffer=False,
            return_barycentrics=False,
        )
        vertices = torch.FloatTensor(model.vertices)
        texture = torch.FloatTensor([
            [16, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]).view(5, 5, 1)
        texture = texture.permute(2, 0, 1)
        render = rasterizer(vertices, texture)

        ans_render = torch.FloatTensor([
            [16, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]).view(5, 5, 1)
        render = render.permute(1, 2, 0)
        self.assertTrue((render - ans_render).norm() < 1e-6)

    def test_forward05(self):
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
                [1, 1],
                [0, 0],
                [1, 0],
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
            return_z_buffer=False,
            return_barycentrics=False,
        )
        vertices = torch.FloatTensor(model.vertices)
        texture = torch.FloatTensor([
            [16, 5, -9, 11, 12],
            [0, 0, 4, 0, 0],
            [10, 0, 8, 5, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 1, 7, 0],
        ]).view(5, 5, 1)
        texture = texture.permute(2, 0, 1)
        render = rasterizer(vertices, texture)

        ans_render = torch.FloatTensor([
            [16, 5, -9, 11, 12],
            [0, 0, 4, 0, 0],
            [10, 0, 8, 5, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 1, 7, 0],
        ]).view(5, 5, 1)
        render = render.permute(1, 2, 0)
        self.assertTrue((render - ans_render).norm() < 1e-6)

    def test_backward_vertices01(self):
        canvas_size = (2, 2)
        model = geom_tools.Mesh(
            vertices=np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, -1],
            ], dtype=np.float32),
            texture_vertices=np.array([
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
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
            return_z_buffer=False,
            return_barycentrics=False,
        )
        texture = torch.FloatTensor([
            [1, 2],
            [3, 4],
        ]).view(2, 2, 1)
        texture = texture.permute(2, 0, 1)

        vertices = torch.FloatTensor(model.vertices).requires_grad_(True)
        render = rasterizer(vertices, texture)
        render = render.permute(1, 2, 0).view(2, 2)

        target = torch.FloatTensor([
            [1, 2],
            [3, 4]
        ])

        diff = render - target
        loss = diff.norm()
        loss.backward()

        ans_vertices_grad = torch.FloatTensor([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        self.assertTrue((vertices.grad - ans_vertices_grad).norm() < 1e-6)

        ans_render = torch.FloatTensor([
            [1, 2],
            [3, 4],
        ])

        self.assertTrue((render - ans_render).norm() < 1e-6)

    def test_backward_vertices02(self):
        canvas_size = (2, 2)
        model = geom_tools.Mesh(
            vertices=np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, -1],
            ], dtype=np.float32),
            texture_vertices=np.array([
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
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
            return_z_buffer=False,
            return_barycentrics=True,
        )
        texture = torch.FloatTensor([
            [2, 2],
            [3, 3],
        ]).view(2, 2, 1)
        texture = texture.permute(2, 0, 1)

        vertices = torch.FloatTensor(model.vertices).requires_grad_(True)
        render, l1l2l3, tri_indices = rasterizer(vertices, texture)
        render = render.permute(1, 2, 0).view(2, 2)

        target = torch.FloatTensor([
            [0, 0],
            [2, 2]
        ])

        diff = render - target
        loss = diff.abs().sum()
        loss.backward()

        ans_vertices_grad = torch.FloatTensor([
            [-2, -3, 0],
            [2, -3, 0],
            [-3, 2, 0],
            [3, 2, 0],
        ])
        # print(vertices.grad)
        # print(ans_vertices_grad.dtype)
        self.assertTrue((vertices.grad - ans_vertices_grad).norm().item() < 1e-6)

        ans_render = torch.FloatTensor([
            [2, 2],
            [3, 3],
        ])

        self.assertTrue((render - ans_render).abs().sum().item() < 1e-6)

    def test_backward_vertices03(self):
        canvas_size = (2, 2)
        model = geom_tools.Mesh(
            vertices=np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, -1],
            ], dtype=np.float32),
            texture_vertices=np.array([
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
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
            return_z_buffer=False,
            return_barycentrics=True,
        )
        texture = torch.FloatTensor([
            [2, 2],
            [3, 3],
        ]).view(2, 2, 1)
        texture = texture.permute(2, 0, 1)

        vertices = torch.FloatTensor(model.vertices).requires_grad_(True)
        render, l1l2l3, tri_indices = rasterizer(vertices, texture)
        render = render.permute(1, 2, 0).view(2, 2)

        target = torch.FloatTensor([
            [0, 0],
            [2, 2]
        ])

        diff = render - target
        loss = diff.abs().sum()
        loss.backward()

        # print("vertices.grad", vertices.grad)

        ans_vertices_grad = -torch.FloatTensor([
            [2, 3, 0],
            [-2, 3, 0],
            [3, -2, 0],
            [-3, -2, 0],
        ])
        self.assertTrue((vertices.grad - ans_vertices_grad).norm() < 1e-6)

        ans_render = torch.FloatTensor([
            [2, 2],
            [3, 3],
        ])

        self.assertTrue((render - ans_render).norm() < 1e-6)

    def test_backward_vertices04(self):
        canvas_size = (4, 4)
        model = geom_tools.Mesh(
            vertices=np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ], dtype=np.float32) * 3,
            texture_vertices=np.array([
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
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
            return_z_buffer=False,
            return_barycentrics=True,
        )
        texture = torch.FloatTensor([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ]).unsqueeze(-1)
        texture = texture.permute(2, 0, 1)

        vertices = torch.FloatTensor(model.vertices).requires_grad_(True)
        render, l1l2l3, tri_indices = rasterizer(vertices, texture)
        render = render.permute(1, 2, 0).squeeze()

        # print(l1l2l3.permute(2, 0, 1))

        target = torch.FloatTensor([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ])

        # print(render)

        diff = render - target
        loss = diff.abs().sum()
        loss.backward()

        ans_vertices_grad = torch.FloatTensor([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        # print(vertices.grad)
        # print(ans_vertices_grad.dtype)
        self.assertTrue((vertices.grad - ans_vertices_grad).norm().item() < 1e-6)

        ans_render = torch.FloatTensor([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ])

        self.assertTrue((render - ans_render).abs().sum().item() < 1e-6)


if __name__ == '__main__':
    unittest.main()
