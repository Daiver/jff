import cv2
import numpy as np
import torch

import geom_tools
import torch_rasterizer


def main():
    canvas_size = (128, 128)
    model = geom_tools.Mesh(
        vertices=np.array([
            [10, 10, 1],
            [50, 10, 1],
            [10, 50, 0],
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

    rasterizer = torch_rasterizer.mk_rasterizer(
        model.triangle_vertex_indices,
        model.triangle_texture_vertex_indices,
        canvas_size)
    vertices = torch.FloatTensor(model.vertices)
    texture = torch.zeros(canvas_size)
    _, z_buffer, _, _ = rasterizer(vertices, texture)

    z_buffer = z_buffer.detach().numpy()

    z_diff = (z_buffer.max() - z_buffer.min())
    print(z_diff)
    z_buffer = (z_buffer - z_buffer.min()) / z_diff
    cv2.imshow("", z_buffer)
    cv2.waitKey()


if __name__ == '__main__':
    main()
