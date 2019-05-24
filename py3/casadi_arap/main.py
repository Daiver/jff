import numpy as np
import casadi
from casadi import SX, Function


def point_to_point_residual(vertex, target):
    return vertex - target


def fan_laplacian(fan_center, fan_neighbours):
    return fan_neighbours - fan_center


def fan_arap_residual(fan_center, fan_neighbours, old_fan_center, old_fan_neighbours):
    res = fan_laplacian(fan_center, fan_neighbours) - fan_laplacian(old_fan_center, old_fan_neighbours)
    return res.reshape((-1, 1))


def fan_rot_arap_residual(fan_rotation, fan_center, fan_neighbours, old_fan_center, old_fan_neighbours):
    old_laplacian = fan_laplacian(old_fan_center, old_fan_neighbours)
    new_laplacian = fan_laplacian(fan_center, fan_neighbours)
    res = fan_rotation @ new_laplacian - old_laplacian
    return res.reshape((-1, 1))


def make_rot_arap_residuals(adjacency, old_vertices_positions, new_vertices_rotation, new_vertices_positions):
    n_vertices = len(adjacency)
    assert n_vertices == old_vertices_positions.size2()
    assert new_vertices_positions.size2() == old_vertices_positions.size2()
    assert new_vertices_positions.size2() * 3 == new_vertices_rotation.size2()

    residuals = []
    for v_ind, ring1_indices in enumerate(adjacency):
        old_fan_center = old_vertices_positions[:, v_ind]
        old_fan_ring1 = old_vertices_positions[:, ring1_indices]

        new_fan_center = new_vertices_positions[:, v_ind]
        new_fan_ring1 = new_vertices_positions[:, ring1_indices]

        fan_rotation = new_vertices_rotation[:, 3 * v_ind: 3 * (v_ind + 1)]

        # print(ring1_indices)
        # print(old_vertices_positions.shape)
        # print(old_fan_center.shape, old_fan_ring1.shape)

        residuals.append(
            fan_rot_arap_residual(fan_rotation, new_fan_center, new_fan_ring1, old_fan_center, old_fan_ring1))
    residuals = casadi.vertcat(*residuals)
    return residuals


def test01():
    vertex = SX.sym("vertex", 3)
    target = SX.sym("target", 3)
    p2p_func = Function("point2point", [vertex, target], [point_to_point_residual(vertex, target)])

    n_vertices = 2
    vertices = SX.sym("vertices", 3, n_vertices)
    targets = SX.sym("targets", 3, n_vertices)
    residuals = []
    for i in range(n_vertices):
        residuals.append(p2p_func(vertices[:, i], targets[:, i]))
    residuals = casadi.vertcat(*residuals)

    print(p2p_func([0, 1, 2], [2, 3, 1]))

    print(residuals)
    print(vertices)
    # print(Function("tmp", [vertices, targets], [residuals])(
    #     np.array([[1, 0, 2], [4, -1, 5]], dtype=np.float32).T,
    #     np.array([[0, 0, 0], [0, -1, 66]], dtype=np.float32).T,
    # ))
    variables = casadi.vertcat(casadi.reshape(vertices, -1, 1))
    jac = casadi.jacobian(residuals, variables)
    jac_f = Function("jac_f", [vertices, targets], [jac])

    num_jac_val = jac_f(
        np.array([[1, 0, 2], [4, -1, 5]], dtype=np.float32).T,
        np.array([[0, 0, 0], [0, -1, 66]], dtype=np.float32).T)

    print(variables)
    print(jac)
    print("jac.nnz()", jac.nnz())
    print("jac.is_dense()", jac.is_dense())
    print(jac_f)
    print(num_jac_val)
    print("num_jac_val.is_dense()", num_jac_val.is_dense())

    print(type(num_jac_val.sparse()))


def main():
    # test01()
    n_vertices = 3

    adjacency = [
        [1, 2],
        [0, 2],
        [0, 1]
    ]
    assert len(adjacency) == n_vertices

    old_vertices = SX.sym("old_vertices", 3, n_vertices)
    vertices = SX.sym("vertices", 3, n_vertices)
    rotations = SX.sym("rotations", 3, 3 * n_vertices)

    arap = make_rot_arap_residuals(adjacency, old_vertices, rotations, vertices)

    residuals = arap

    variables = casadi.vertcat(
        # rotations.reshape((-1, 1)),
        vertices.reshape((-1, 1))
    )
    jac = casadi.jacobian(residuals, variables)
    print(jac)


if __name__ == '__main__':
    main()
