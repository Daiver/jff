import numpy as np
import casadi
from casadi import SX, Function


def point_to_point_residual(vertex, target):
    return vertex - target


def arap_residual(fan_center, fan_neighbours):
    return (fan_neighbours - fan_center).reshape((-1, 1))


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
    n_vertices = 5
    vertices = SX.sym("vertices", 3, n_vertices)
    fan_center = vertices[:, 0]
    ring1 = vertices[:, 1:]

    print(vertices)
    print(fan_center)
    print(ring1)
    print(arap_residual(fan_center, ring1))


if __name__ == '__main__':
    main()
