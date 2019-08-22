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
    return point - target_point


def mk_unrolled_gauss_newton(residuals_func, jacobian_func, x0, n_iters, dumping_const):
    x = x0

    eye = np.eye(x0.shape[0])

    for i in range(n_iters):
        residuals = residuals_func(x)
        jacobian = jacobian_func(x)

        gradient = jacobian.T @ residuals
        hessian = jacobian.T @ jacobian
        hessian = hessian + dumping_const * eye

        step = casadi.inv(hessian) @ gradient
        x = x - step
    residuals = residuals_func(x)
    jacobian = jacobian_func(x)
    return x, 0.5 * residuals.T @ residuals, jacobian.T @ residuals


def main():
    a, b, c = SX(1), SX(0), SX(0)
    a = SX.sym("a")
    target = SX([2, 2])
    # target = SX([1, 1])
    quadratic_sx = mk_quadratic(a, b, c)
    x_sx = SX.sym("x")
    residuals = point_to_curve_residual(quadratic_sx, x_sx, target)
    jacobian = casadi.jacobian(residuals, x_sx)

    residuals_func = Function("residuals_func", [x_sx], [residuals])
    jacobian_func = Function("jacobian_func", [x_sx], [jacobian])

    dump_const = 0.1
    gauss_newton = mk_unrolled_gauss_newton(residuals_func, jacobian_func, x_sx, 10, dump_const)
    gauss_newton_func = Function("gauss_newton_func", [x_sx, a], gauss_newton)

    print(casadi.jacobian(gauss_newton_func(x_sx, a)[0], a))

    variables = [0]
    print(gauss_newton_func(variables, [1]))


def main1():
    a, b, c = SX(1), SX(0), SX(0)
    target = SX([2, 2])
    # target = SX([1, 1])
    quadratic_sx = mk_quadratic(a, b, c)
    x_sx = SX.sym("x")
    residuals = point_to_curve_residual(quadratic_sx, x_sx, target)
    jacobian = casadi.jacobian(residuals, x_sx)
    hessian, gradient = casadi.hessian(0.5 * residuals.T @ residuals, x_sx)
    # print(hessian)

    residuals_func = Function("residuals_func", [x_sx], [residuals])
    jacobian_func = Function("jacobian_func", [x_sx], [jacobian])
    # hessian_func = Function("hessian_func", [x_sx], [hessian])

    dump_const = 0.1
    gauss_newton = mk_unrolled_gauss_newton(residuals_func, jacobian_func, x_sx, 10, dump_const)
    gauss_newton_func = Function("gauss_newton_func", [x_sx], [gauss_newton])

    variables = [0]
    for i in range(10):
        residuals_val = residuals_func(variables).toarray()
        jacobian_val = jacobian_func(variables).toarray()
        loss_val = residuals_val.T.dot(residuals_val)

        gradient_val = jacobian_val.T @ residuals_val
        hessian_val = jacobian_val.T @ jacobian_val
        # hessian_val = hessian_func(variables)
        hessian_val += dump_const * np.eye(hessian_val.shape[0])
        step = np.linalg.solve(hessian_val, gradient_val)

        print(f"{i} loss {loss_val} var = {variables}, \grad_norm = {np.linalg.norm(gradient_val)}")
        variables -= step

    print(variables)
    print(gauss_newton_func(variables))


if __name__ == '__main__':
    main()
