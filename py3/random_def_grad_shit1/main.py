import casadi
from casadi import DM, SX, MX, Function


def shape_matrix2d_from_vertices(triangle_vertices: SX) -> SX:
    assert triangle_vertices.size1() == 2
    assert triangle_vertices.size2() == 3

    col0 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    col1 = triangle_vertices[:, 2] - triangle_vertices[:, 0]

    return casadi.horzcat(col0, col1)


def deformation_gradient2d(triangle_vertices: SX, old_triangle_vertices: SX) -> SX:
    assert triangle_vertices.shape == old_triangle_vertices.shape
    new_shape_mat = shape_matrix2d_from_vertices(triangle_vertices)
    old_shape_mat = shape_matrix2d_from_vertices(old_triangle_vertices)
    return new_shape_mat @ casadi.inv(old_shape_mat)


def main():
    vertices = SX.sym("v", 2, 3)
    old_vertices = DM([1, 2, 3, 5, 8, 4]).reshape((2, 3))
    res = deformation_gradient2d(vertices, old_vertices)
    print(res.shape)
    print(res)


if __name__ == '__main__':
    main()
