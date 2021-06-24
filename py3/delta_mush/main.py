from typing import List
import numpy as np
import geom_tools


def laplacian_smoothing_one_step(
        vertices: np.ndarray,
        adjacency_table: List[List[int]],
        degree: float
) -> np.ndarray:
    assert 0 <= degree <= 1
    n_vertices = len(adjacency_table)
    assert n_vertices == len(vertices)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    res = vertices.copy()
    for vertex_ind in range(n_vertices):
        average_vertex = np.zeros(3, dtype=np.float)
        for adj_vertex_ind in adjacency_table[vertex_ind]:
            average_vertex += vertices[adj_vertex_ind]
        average_vertex *= 1.0 / len(adjacency_table[vertex_ind])
        res[vertex_ind] = (1.0 - degree) * vertices[vertex_ind] + degree * average_vertex

    return res


def laplacian_smoothing(vertices: np.ndarray) -> np.ndarray:
    pass


def main():
    vertices = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32)
    triangle_indices = [
        [0, 1, 2]
    ]
    adj_table = geom_tools.adjacency_tools.vertices_adjacency_from_polygon_vertex_indices(triangle_indices)
    res = laplacian_smoothing_one_step(vertices, adj_table, 0.5)
    pass


if __name__ == "__main__":
    main()
