import numpy as np
import casadi
from casadi import SX, Function


def mk_point_to_point_residual(vertex, target):
    return vertex - target


# def mk_


def main():
    vertex = SX.sym("vertex", 3)
    target = SX.sym("target", 3)
    p2p_func = Function("point2point", [vertex, target], [mk_point_to_point_residual(vertex, target)])

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
    print(variables)
    jac = casadi.jacobian(residuals, variables)
    print(jac)
    jac_f = Function("jac_f", [vertices, targets], [jac])
    print(jac_f)

    num_jac_val = jac_f(
        np.array([[1, 0, 2], [4, -1, 5]], dtype=np.float32).T,
        np.array([[0, 0, 0], [0, -1, 66]], dtype=np.float32).T)
    print(num_jac_val)

    print(type(num_jac_val.toarray()))


if __name__ == '__main__':
    main()
