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


def triangle_vertices_from_vertices(vertices: SX, triangle_inds: [int]) -> SX:
    assert len(triangle_inds) == 3
    return casadi.horzcat(vertices[:, triangle_inds[0]], vertices[:, triangle_inds[1]], vertices[:, triangle_inds[2]])


def deformation_gradients2d(
        vertices: SX,
        old_vertices: SX,
        triangle_vertex_indices: [[int]]
) -> SX:
    assert vertices.shape == old_vertices.shape
    assert vertices.size1() == 2
    res = []
    for triangle_inds in triangle_vertex_indices:
        triangle_vertices = triangle_vertices_from_vertices(vertices, triangle_inds)
        old_triangle_vertices = triangle_vertices_from_vertices(old_vertices, triangle_inds)
        res.append(deformation_gradient2d(triangle_vertices, old_triangle_vertices))
    return casadi.horzcat(*res)


def activation_matrix2d_from_activations(muscle_activations: SX) -> SX:
    res = casadi.horzcat(muscle_activations[0], 0, 0, muscle_activations[1])
    return res.reshape((2, 2))


def passive_energy_for_triangle(
        triangle_rotation: SX,
        def_grad: SX,
        triangle_area: float,
        weight_corotation: float,
        weight_volume_preserve: float
) -> SX:
    corotated_energy = weight_corotation * casadi.norm_2((def_grad - triangle_rotation).reshape((-1, 1)))
    volume_preserve_energy = weight_volume_preserve * (casadi.det(def_grad) - 1)**2

    return triangle_area * (corotated_energy + volume_preserve_energy)


def active_energy_for_triangle(
        triangle_rotation: SX,
        def_grad: SX,
        target_grad: SX,
        triangle_area: float,
        weight_corotation: float,
        weight_volume_preserve: float
) -> SX:
    assert target_grad.shape == def_grad.shape
    corotated_energy = weight_corotation * casadi.norm_2((def_grad - triangle_rotation @ target_grad).reshape((-1, 1)))
    volume_preserve_energy = weight_volume_preserve * (casadi.det(def_grad) - casadi.det(target_grad))**2

    return triangle_area * (corotated_energy + volume_preserve_energy)


def energy_for_triangle_mesh(
        vertices_positions: SX,
        old_vertices_positions: SX,
        rotations_per_passive_triangle: SX,
        rotations_per_active_triangle: SX,
        activations: SX,
        passive_triangle_vertex_indices: [[int]],
        active_triangle_vertex_indices: [[int]],
        weight_corotation: float,
        weight_volume_preserve: float
) -> SX:
    passive_def_grads = deformation_gradients2d(
        vertices_positions,
        old_vertices_positions,
        passive_triangle_vertex_indices
    )
    active_def_grads = deformation_gradients2d(
        vertices_positions,
        old_vertices_positions,
        active_triangle_vertex_indices
    )

    res = 0
    for tri_ind, tri_vertex_inds in enumerate(passive_triangle_vertex_indices):
        rotation = rotations_per_passive_triangle[:, 2 * tri_ind: 2 * (tri_ind + 1)]
        def_grad = passive_def_grads[:, 2 * tri_ind: 2 * (tri_ind + 1)]
        res += passive_energy_for_triangle(
            rotation, def_grad,
            triangle_area=1,
            weight_corotation=weight_corotation, weight_volume_preserve=weight_volume_preserve
        )

    assert activations.size1() == 2
    for tri_ind, tri_vertex_inds in enumerate(active_triangle_vertex_indices):
        rotation = rotations_per_active_triangle[:, 2 * tri_ind: 2 * (tri_ind + 1)]
        def_grad = active_def_grads[:, 2 * tri_ind: 2 * (tri_ind + 1)]
        local_activations = activations[:, tri_ind]
        target_grad = activation_matrix2d_from_activations(local_activations)
        res += active_energy_for_triangle(
            rotation, def_grad, target_grad=target_grad,
            triangle_area=1,
            weight_corotation=weight_corotation, weight_volume_preserve=weight_volume_preserve
        )

    return res


def main():
    vertices = SX.sym("v", 2, 3)
    rotation = SX.sym("r", 2, 2)
    activations = SX.sym("a", 2)
    old_vertices = DM([1, 2, 3, 5, 8, 4]).reshape((2, 3))

    res = energy_for_triangle_mesh(
        vertices_positions=vertices, old_vertices_positions=old_vertices,
        rotations_per_passive_triangle=rotation,
        rotations_per_active_triangle=rotation,
        activations=activations,
        passive_triangle_vertex_indices=[[0, 1, 2]],
        active_triangle_vertex_indices=[[0, 1, 2]],
        weight_corotation=1, weight_volume_preserve=1
    )
    print(res)


if __name__ == '__main__':
    main()
