import numpy as np
import matplotlib.pyplot as plt
import casadi
from casadi import SX, Function

from pygauss_newton import gauss_newton


def mk_quadratic(a, b, c):
    def inner_quadratic(x):
        # return a * x * x + b * x + c
        return a * x
    return inner_quadratic


def mk_point_to_curve_residual(curve_function, current_x, target_point):
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
    loss = 0.5 * residuals.T @ residuals
    jacobian = jacobian_func(x)
    hessian = jacobian.T @ residuals
    return x, residuals, loss, hessian


def main():
    a = SX.sym("a")
    b, c = SX(0), SX(0)
    # target = SX([2, 2])
    # target = SX([3, 2])
    targets = [
        [1, 0],
        # [0, 0],
        [0, 1],
    ]
    # targets = SX(targets)
    # target = SX([1, 1])
    quadratic_sx = mk_quadratic(a, b, c)

    unroll_dump_const = 0.1
    x_sxs = SX.sym("x", len(targets))
    residuals_to_point = []
    unrolled_functions = []
    for i, target in enumerate(targets):
        x_sx = x_sxs[i]
        point_to_curve_residual = mk_point_to_curve_residual(quadratic_sx, x_sx, SX(target))
        point_to_curve_jacobian = casadi.jacobian(point_to_curve_residual, x_sx)

        point_to_curve_residuals_func = Function(f"point_to_curve_residuals_func{i}", [x_sx], [point_to_curve_residual])
        point_to_curve_jacobian_func = Function(f"point_to_curve_jacobian_func{i}", [x_sx], [point_to_curve_jacobian])

        unrolled = mk_unrolled_gauss_newton(
            point_to_curve_residuals_func, point_to_curve_jacobian_func, x_sx, 15, unroll_dump_const)
        unrolled_func = Function("unrolled_func", [x_sx, a], unrolled)
        unrolled_functions.append(unrolled_func)
        residual_to_point = unrolled[1]
        residuals_to_point.append(residual_to_point)

    residuals_to_point = casadi.vertcat(*residuals_to_point)
    residuals_to_point_func = Function("residuals_to_point_func", [x_sxs, a], [residuals_to_point])

    point_to_curve_jacobian = casadi.jacobian(residuals_to_point_func(x_sxs, a), a)
    point_to_curve_jacobian_func = Function("residuals_to_point_jac", [x_sxs, a], [point_to_curve_jacobian])

    print(residuals_to_point_func)

    dump_const = 0.0001

    vars = np.array([[1.0]])
    xs = np.array([0.0] * len(targets))
    for i in range(20):
        residuals_val = residuals_to_point_func(xs, vars)
        jacobian_val = point_to_curve_jacobian_func(xs, vars)

        loss_val = 0.5 * residuals_val.T @ residuals_val
        gradient_val = jacobian_val.T @ residuals_val
        hessian_val = jacobian_val.T @ jacobian_val
        hessian_val += dump_const * np.eye(hessian_val.shape[0])

        step = -np.linalg.solve(hessian_val, gradient_val)

        print(f"{i}, loss = {loss_val}, grad_norm = {np.linalg.norm(gradient_val)}")
        vars += step
        for x, unrolled_func in zip(xs, unrolled_functions):
            print(vars, unrolled_func(x, vars)[0])
    print(vars)


if __name__ == '__main__':
    main()
