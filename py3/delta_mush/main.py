from typing import List
import numpy as np
import geom_tools
from copy import deepcopy


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
        average_vertex = np.zeros(3, dtype=np.float32)
        for adj_vertex_ind in adjacency_table[vertex_ind]:
            average_vertex += vertices[adj_vertex_ind]
        average_vertex *= 1.0 / len(adjacency_table[vertex_ind])
        res[vertex_ind] = (1.0 - degree) * vertices[vertex_ind] + degree * average_vertex

    return res


def laplacian_smoothing(
        vertices: np.ndarray,
        adjacency_table: List[List[int]],
        degree: float,
        n_iters: int
) -> np.ndarray:
    res = vertices.copy()
    for iteration in range(n_iters):
        res = laplacian_smoothing_one_step(res, adjacency_table, degree)
    return res


def normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    assert length > 0
    return vector / length


def compute_basis_for_fan(
        vertices: np.ndarray,
        normals: np.ndarray,
        adjacency_table: List[List[int]],
        vertex_ind: int
) -> np.ndarray:
    assert vertices.shape == normals.shape
    adj_indices = adjacency_table[vertex_ind]
    assert len(adj_indices) > 0
    normal = normals[vertex_ind]
    first_edge = vertices[adj_indices[0]] - vertices[vertex_ind]
    first_axis = normalize(np.cross(normal, first_edge))
    second_axis = normalize(np.cross(first_axis, normal))
    return np.vstack((normal, first_axis, second_axis))


def compute_basis_for_mesh(
        vertices: np.ndarray,
        normals: np.ndarray,
        adjacency_table: List[List[int]],
) -> List[np.ndarray]:
    assert vertices.shape == normals.shape
    n_vertices = vertices.shape[0]
    res = [
        compute_basis_for_fan(vertices, normals, adjacency_table, vertex_ind)
        for vertex_ind in range(n_vertices)
    ]
    return res


def extract_deltas_from_mesh(
        vertices_to_extract_deltas: np.ndarray,
        basis_centers: np.ndarray,
        basis_per_vertex: List[np.ndarray],
) -> np.ndarray:
    assumed_shape = vertices_to_extract_deltas.shape
    n_vertices = assumed_shape[0]
    assert basis_centers.shape == assumed_shape
    assert len(basis_per_vertex) == n_vertices
    res = [
        np.linalg.inv(basis) @ (vertex - center)
        for vertex, center, basis in zip(vertices_to_extract_deltas, basis_centers, basis_per_vertex)
    ]
    return np.array(res)


def apply_deltas(
        deltas: np.ndarray,
        basis_centers: np.ndarray,
        basis_per_vertex: List[np.ndarray],
) -> np.ndarray:
    assumed_shape = deltas.shape
    n_vertices = assumed_shape[0]
    assert basis_centers.shape == assumed_shape
    assert len(basis_per_vertex) == n_vertices
    res = [
        basis @ delta + center
        for delta, center, basis in zip(deltas, basis_centers, basis_per_vertex)
    ]
    return np.array(res)


def main():
    path_to_src = "/home/daiver/src.obj"
    path_to_dst = "/home/daiver/dst.obj"
    path_to_res = "/home/daiver/res.obj"

    smooth_degree = 0.5
    smooth_iters = 100

    geom_source = geom_tools.load(path_to_src)
    geom_target = geom_tools.load(path_to_dst)
    polygon_vertex_indices = geom_source.polygon_vertex_indices
    triangle_vertex_indices = geom_source.triangle_vertex_indices
    assert polygon_vertex_indices == geom_target.polygon_vertex_indices
    adj_table = geom_tools.adjacency_tools.vertices_adjacency_from_polygon_vertex_indices(polygon_vertex_indices)
    vertices_src = geom_source.vertices
    vertices_dst = geom_target.vertices

    vertices_src_smooth = laplacian_smoothing(vertices_src, adj_table, smooth_degree, smooth_iters)
    vertices_dst_smooth = laplacian_smoothing(vertices_dst, adj_table, smooth_degree, smooth_iters)
    normals_src_smooth = geom_tools.compute_vertices_normals_from_triangles(vertices_src_smooth, triangle_vertex_indices)
    normals_dst_smooth = geom_tools.compute_vertices_normals_from_triangles(vertices_dst_smooth, triangle_vertex_indices)

    basis_src = compute_basis_for_mesh(vertices_src_smooth, normals_src_smooth, adj_table)
    basis_dst = compute_basis_for_mesh(vertices_dst_smooth, normals_dst_smooth, adj_table)

    deltas = extract_deltas_from_mesh(vertices_src, vertices_src_smooth, basis_src)
    res_vertices = apply_deltas(deltas, vertices_dst_smooth, basis_dst)
    geom_res = deepcopy(geom_source)
    geom_res.vertices = res_vertices
    geom_res.normals = geom_tools.compute_vertices_normals_from_triangles(geom_res.vertices, triangle_vertex_indices)
    geom_tools.save(geom_res, path_to_res)


if __name__ == "__main__":
    main()
