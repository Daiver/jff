import numpy as np

import geom_tools
import torch_rasterizer


def main():
    canvas_size = (128, 128)
    model = geom_tools.Mesh(
        vertices=np.array([
            [10, 10, 0],
            [50, 10, 0],
            [10, 50, 0],
        ], dtype=np.float32),
        polygon_vertex_indices=[
            [0, 1, 2]
        ],
        texture_vertices=np.array([
            [0, 0],
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


if __name__ == '__main__':
    main()
