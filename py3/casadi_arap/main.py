import numpy as np
import casadi
from casadi import SX, Function

from gaussnewton import perform_gauss_newton


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


def rigid_residual(rotation):
    assert rotation.shape == (3, 3)
    rtr = rotation.T @ rotation
    eye = SX.eye(3)
    return (rtr - eye).reshape((-1, 1))


def make_rigid_residuals(rotations):
    assert rotations.shape[0] == 3
    assert rotations.shape[1] % 3 == 0
    n_rots = rotations.shape[1] // 3
    residuals = []
    for v_ind in range(n_rots):
        rotation = rotations[:, 3 * v_ind: 3 * (v_ind + 1)]
        residuals.append(rigid_residual(rotation))

    residuals = casadi.vertcat(*residuals)
    return residuals


def main():    
    vertices_val = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
    ], dtype=np.float32)

    adjacency = [
        [1, 2, 3, 4],
        [0, 2, 4],
        [0, 1, 3],
        [0, 2, 4],
        [0, 1, 3]
    ]

    targets_val = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [1, 0, 0]
    ], dtype=np.float32)

    target_to_base_indices = [
        1, 2, 4
    ]

    n_vertices = vertices_val.shape[0]
    n_targets = targets_val.shape[0]
    assert len(adjacency) == n_vertices
    assert len(target_to_base_indices) == n_targets

    old_vertices = SX.sym("old_vertices", 3, n_vertices)
    targets = SX.sym("targets", 3, n_targets)

    vertices = SX.sym("vertices", 3, n_vertices)
    rotations = SX.sym("rotations", 3, 3 * n_vertices)

    vertices_to_fit = vertices[:, target_to_base_indices]

    data = (vertices_to_fit - targets).reshape((-1, 1))
    arap = make_rot_arap_residuals(adjacency, old_vertices, rotations, vertices)
    rigid = make_rigid_residuals(rotations)

    residuals = casadi.vertcat(
        data,
        arap,
        rigid
    )

    variables = casadi.vertcat(
        rotations.reshape((-1, 1)),
        vertices.reshape((-1, 1))
    )
    jac = casadi.jacobian(residuals, variables)
    print("jac.shape", jac.shape, "jac.nnz()", jac.nnz())

    # print(residuals)
    # print(jac)

    residual_func = Function("res_func", [variables, old_vertices, targets], [residuals])
    jac_func = Function("jac_func", [variables, old_vertices, targets], [jac])

    def compute_residuals_and_jac(x):
        residuals_val = residual_func(x, vertices_val.T, targets_val.T).toarray()
        jacobian_val = jac_func(x, vertices_val.T, targets_val.T).sparse()
        # print(jacobian_val)
        return residuals_val, jacobian_val

    init_rot = np.hstack([np.eye(3)] * n_vertices)
    init_vertices = vertices_val.T
    init_vars = np.hstack((
        init_rot.reshape(-1),
        init_vertices.reshape(-1)
    ))
    print(init_vars)
    print(perform_gauss_newton(init_vars, compute_residuals_and_jac, 1))


if __name__ == '__main__':
    main()
