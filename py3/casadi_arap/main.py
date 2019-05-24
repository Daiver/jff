import numpy as np

from scipy.spatial.transform import Rotation


import casadi
from casadi import SX, Function

import arap
from gaussnewton import perform_gauss_newton


def deform(
    vertices_val,
    adjacency,
    target_to_base_indices,
    targets_val,
):
    n_vertices = vertices_val.shape[1]
    n_targets = targets_val.shape[1]

    assert vertices_val.shape[0] == 3
    assert targets_val.shape[0] == 3
    assert len(adjacency) == n_vertices
    assert len(target_to_base_indices) == n_targets

    old_vertices = SX.sym("old_vertices", 3, n_vertices)
    targets = SX.sym("targets", 3, n_targets)

    vertices = SX.sym("vertices", 3, n_vertices)
    rotations = SX.sym("rotations", 3, 3 * n_vertices)

    vertices_to_fit = vertices[:, target_to_base_indices]

    data = (vertices_to_fit - targets).reshape((-1, 1))
    arap_residual = arap.make_rot_arap_residuals(adjacency, old_vertices, rotations, vertices)
    rigid = arap.make_rigid_residuals(rotations)

    residuals = casadi.vertcat(
        data,
        arap_residual,
        10.0 * rigid
    )

    variables = casadi.vertcat(
        rotations.reshape((-1, 1)),
        vertices.reshape((-1, 1))
    )
    jac = casadi.jacobian(residuals, variables)
    print("jac.shape", jac.shape, "jac.nnz()", jac.nnz())

    fixed_values = [
        old_vertices,
        targets,
    ]

    residual_func = Function("res_func", [variables] + fixed_values, [residuals])
    jac_func = Function("jac_func", [variables] + fixed_values, [jac])

    def compute_residuals_and_jac(x):
        residuals_val = residual_func(x, vertices_val, targets_val).toarray()
        jacobian_val = jac_func(x, vertices_val, targets_val).sparse()
        return residuals_val, jacobian_val

    init_rot = np.hstack([np.eye(3)] * n_vertices)
    init_vertices = vertices_val
    init_vars = np.hstack((
        init_rot.T.reshape(-1),
        init_vertices.reshape(-1)
    ))

    res = perform_gauss_newton(init_vars, compute_residuals_and_jac, 10)
    new_vertices = res[9 * n_vertices:].reshape(-1, 3).T
    return new_vertices


def main():
    np.set_printoptions(threshold=np.inf, linewidth=500)
    vertices_val = np.array([
        [0, 0, 1],
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

    target_to_base_indices = list(range(len(vertices_val)))
    targets_val = vertices_val.copy()
    rot_mat = Rotation.from_euler('z', 90, degrees=True).as_dcm()
    print(rot_mat)
    targets_val = targets_val @ rot_mat.T

    # targets_val = np.array([
    #     [0, 0, 2],
    #     [0, 1, 1],
    #     [-1, 0, 1],
    #     # [-1, 0, 1],
    #     # [0, -1, 1],
    # ], dtype=np.float32)
    #
    # target_to_base_indices = [
    #     0,
    #     1,
    #     2,
    # ]

    new_vertices = deform(
        vertices_val.T, adjacency,
        target_to_base_indices, targets_val.T
    )
    print(new_vertices)


if __name__ == '__main__':
    main()
