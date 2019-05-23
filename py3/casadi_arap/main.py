import numpy as np
import casadi
from casadi import SX, Function


def mk_point_to_point_residual(vertex, target):
    return vertex - target


def main():
    vertex = SX.sym("vertex", 3)
    target = SX.sym("target", 3)
    p2p_func = Function("point2point", [vertex, target], [mk_point_to_point_residual(vertex, target)])

    n_vertices = 2
    vertices = SX.sym("vertices", n_vertices, 3)
    targets = SX.sym("targets", n_vertices, 3)
    residuals = []
    for i in range(n_vertices):
        residuals.append(p2p_func(vertices[i, :], targets[i, :]))
    residuals = casadi.vertcat(*residuals)

    print(p2p_func([0, 1, 2], [2, 3, 1]))

    print(residuals)

    print(Function("tmp", [vertices, targets], [residuals])(
        np.array([[1, 0, 2], [4, -1, 5]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 66]], dtype=np.float32),
    ))


if __name__ == '__main__':
    main()
