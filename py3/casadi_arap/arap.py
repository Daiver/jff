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

        residuals.append(
            fan_rot_arap_residual(fan_rotation, new_fan_center, new_fan_ring1, old_fan_center, old_fan_ring1))
    residuals = casadi.vertcat(*residuals)
    return residuals


def rigid_residual(rotation):
    assert rotation.shape == (3, 3)
    rtr = rotation.T @ rotation
    # print(rtr)
    eye = SX.eye(3)
    res = rtr - eye
    return res.reshape((-1, 1))


def make_rigid_residuals(rotations):
    assert rotations.shape[0] == 3
    assert rotations.shape[1] % 3 == 0
    n_rots = rotations.shape[1] // 3
    residuals = []

    rotation = SX.sym("rotation", 3, 3)
    rigid_func = Function("rigid_func", [rotation], [rigid_residual(rotation)])

    for v_ind in range(n_rots):
        rotation = rotations[:, 3 * v_ind: 3 * (v_ind + 1)]
        residuals.append(rigid_func(rotation))

    residuals = casadi.vertcat(*residuals)
    return residuals

