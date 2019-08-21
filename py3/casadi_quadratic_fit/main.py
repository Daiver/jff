import numpy as np
import matplotlib.pyplot as plt
import casadi
from casadi import SX, Function


def mk_quadratic(a, b, c):
    def inner_quadratic(x):
        return a * x * x + b * x + c
    return inner_quadratic


def point_to_curve_residual(curve_function, current_x, target_point):
    assert target_point.shape == (2, 1)
    y = curve_function(current_x)
    point = casadi.vcat((current_x, y))
    return casadi.norm_2(point - target_point)


def main():
    a, b, c = SX(1), SX(0), SX(0)
    target = SX([2, 2])
    quadratic_sx = mk_quadratic(a, b, c)
    x_sx = SX.sym("x")
    residuals = point_to_curve_residual(quadratic_sx, x_sx, target)
    jacobian = casadi.jacobian(residuals, x_sx)
    hessian, gradient = casadi.hessian(0.5 * residuals @ residuals, x_sx)
    # print(hessian)

    residuals_func = Function("residuals_func", [x_sx], [residuals])
    jacobian_func = Function("jacobian_func", [x_sx], [jacobian])
    hessian_func = Function("hessian_func", [x_sx], [hessian])

    variables = [0]
    dump_contant = 1.0
    for i in range(30):
        residuals_val = residuals_func(variables).toarray()
        jacobian_val = jacobian_func(variables).toarray()
        loss_val = residuals_val.dot(residuals_val)

        gradient_val = jacobian_val.T @ residuals_val
        hessian_val = jacobian_val.T @ jacobian_val
        hessian_val = hessian_func(variables)
        # hessian_val += dump_contant * np.eye(hessian_val.shape[0])
        step = np.linalg.solve(hessian_val, gradient_val)

        print(f"{i} loss {loss_val} var = {variables}, \grad_norm = {np.linalg.norm(gradient_val)}")
        variables -= step

    print(variables)


if __name__ == '__main__':
    main()
