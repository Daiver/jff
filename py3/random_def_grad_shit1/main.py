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


def activation_matrix_from_activations(muscle_activations: SX) -> SX:
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


def main():
    vertices = SX.sym("v", 2, 3)
    rotation = SX.sym("r", 2, 2)
    activations = SX.sym("a", 2)
    old_vertices = DM([1, 2, 3, 5, 8, 4]).reshape((2, 3))

    def_grad = deformation_gradient2d(vertices, old_vertices)
    target_grad = activation_matrix_from_activations(activations)
    # res = passive_energy_for_triangle(rotation, def_grad, 1, 1, 3)
    res = active_energy_for_triangle(rotation, def_grad, target_grad, 1, 1, 3)
    print(res)


if __name__ == '__main__':
    main()
