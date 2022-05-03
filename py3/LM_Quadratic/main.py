from typing import Callable, Tuple

import casadi
import numpy as np
import scipy.optimize
from casadi import DM, SX, Function


def residual_func_and_jac_from_casadi(
        residuals: SX, variables_stacked: SX
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    jac = casadi.jacobian(residuals, variables_stacked)
    residuals_cas_func = Function("residuals_func", [variables_stacked], [residuals])
    jacobian_cas_func = Function("jacobian_func", [variables_stacked], [jac])

    def res_residuals_func(x: np.ndarray):
        return residuals_cas_func(DM(x)).toarray()#.reshape(-1)

    def res_jacobian_func(x: np.ndarray):
        return jacobian_cas_func(DM(x)).toarray()

    return res_residuals_func, res_jacobian_func


def error_grad_hess_func_casadi_residuals(
        residuals: SX, variables_stacked: SX
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    error = 0.5 * casadi.sumsqr(residuals)
    # grad = casadi.gradient(error, variables_stacked)
    hess, grad = casadi.hessian(error, variables_stacked)
    err_cas_func = Function("err_cas_func", [variables_stacked], [error])
    grad_cas_func = Function("grad_func", [variables_stacked], [grad])
    hess_cas_func = Function("hess_func", [variables_stacked], [hess])

    def res_error_func(x: np.ndarray):
        return err_cas_func(DM(x))[0, 0]

    def res_grad_func(x: np.ndarray):
        return grad_cas_func(DM(x)).toarray().reshape(-1)

    def res_hess_func(x: np.ndarray):
        return hess_cas_func(DM(x)).toarray()

    return res_error_func, res_grad_func, res_hess_func


def gn_error_grad_hess_func_casadi_residuals(
        residuals: SX, variables_stacked: SX
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    error = 0.5 * casadi.sumsqr(residuals)
    jac = casadi.jacobian(residuals, variables_stacked)
    err_cas_func = Function("err_cas_func", [variables_stacked], [error])
    grad_cas_func = Function("grad_func", [variables_stacked], [jac.T @ residuals])
    hess_cas_func = Function("hess_func", [variables_stacked], [jac.T @ jac])

    def res_error_func(x: np.ndarray):
        return err_cas_func(DM(x))[0, 0]

    def res_grad_func(x: np.ndarray):
        return grad_cas_func(DM(x)).toarray().reshape(-1)

    def res_hess_func(x: np.ndarray):
        return hess_cas_func(DM(x)).toarray()

    return res_error_func, res_grad_func, res_hess_func


def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(a, b, rcond=None)[0].reshape(-1, 1)


def line_search_for_residuals(
        residuals_func: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        d: np.ndarray
) -> float:
    def fun(a: float) -> float:
        residuals = residuals_func(x0 + a * d)
        return residuals.T @ residuals
    return scipy.optimize.minimize_scalar(fun=fun, bracket=(0, 1))["x"][0, 0]


def perform_lm(
        residuals_func: Callable[[np.ndarray], np.ndarray],
        jacobian_func: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray
):
    x = x0.copy()
    n_vars = len(x0)
    assert x0.shape == (n_vars, 1)
    n_iters = 15
    print(x.T)
    for i in range(n_iters):
        residuals = residuals_func(x)
        jacobian = jacobian_func(x)
        n_residuals = len(residuals)
        assert residuals.shape == (n_residuals, 1)
        assert jacobian.shape == (n_residuals, n_vars)

        jtr = jacobian.T @ residuals
        mu = np.linalg.norm(residuals, ord=None)
        # mu = np.linalg.norm(residuals.reshape(-1), ord=(1 + 1/(i+1)))
        mu = 0
        diag = np.eye(len(x))

        jtj = jacobian.T @ jacobian + diag * mu
        step = solve(jtj, -jtr)
        x = x + step

        # jtr2 = jacobian.T @ residuals_func(x)
        # step2 = solve(jtj, -jtr2)
        # step2_len = 1
        # step2_len = line_search_for_residuals(residuals_func, x, step2)
        # print(step2_len)
        # x = x + step2_len * step2

        print(
            f"{i + 1}/{n_iters} "
            f"f={(0.5 * residuals.T @ residuals)[0, 0]}, "
            f"|g| = {np.linalg.norm(jtr)}, "
            f"|dx| = {np.linalg.norm(step)}, "
            f"mu = {mu}"
        )
        # print(x)
    return x


def main1():
    residuals_func = lambda x: x**2 - 1
    jacobian_func = lambda x: (2 * x).reshape(1, 1)
    perform_lm(residuals_func, jacobian_func, x0=np.array([0.001]))


def main2():
    x = SX.sym("x", 2)
    residuals = casadi.sumsqr(x) - 1

    residuals_func, jacobian_func = residual_func_and_jac_from_casadi(residuals, x)
    x0 = np.array([0.001, -0.001])
    perform_lm(residuals_func, jacobian_func, x0=x0.reshape(-1, 1))


def main():
    base_vertices = DM([
        [1, 0],
        [0, 1],
    ])

    target_vertices = DM([
        [0, -1],
        [6, 5],
    ])

    rotation = SX.sym("rotation", 2, 2)
    translation = SX.sym("translation", 2)
    residuals_rot = rotation.T @ rotation - DM.eye(rotation.size1())
    residuals_fit = rotation @ base_vertices - target_vertices
    residuals_fit[0, :] += translation[0]
    residuals_fit[1, :] += translation[1]

    x = casadi.veccat(rotation.reshape((-1, 1)), translation)
    residuals = casadi.veccat(residuals_rot.reshape((-1, 1)), residuals_fit.reshape((-1, 1)))

    residuals_func, jacobian_func = residual_func_and_jac_from_casadi(residuals, x)
    x0 = np.array([
        1, 0,
        0, 1,
        -5,
        0
    ])
    print(perform_lm(residuals_func, jacobian_func, x0=x0.reshape(-1, 1)).T)

    if False:
        method = "trust-constr"
        max_iter = 8
        err, grad, hess = error_grad_hess_func_casadi_residuals(residuals, x)
        scipy.optimize.minimize(err, x0, jac=grad, hess=hess, method=method, options={"disp": True, "maxiter": max_iter})
        err, grad, hess = gn_error_grad_hess_func_casadi_residuals(residuals, x)
        scipy.optimize.minimize(err, x0, jac=grad, hess=hess, method=method, options={"disp": True, "maxiter": max_iter})

    if False:
        from cubic_newton import cubic_reg
        # err, grad, hess = error_grad_hess_func_casadi_residuals(residuals, x)
        err, grad, hess = gn_error_grad_hess_func_casadi_residuals(residuals, x)
        # solver = cubic_reg.CubicRegularization(x0, f=err, gradient=grad, hessian=hess)
        solver = cubic_reg.AdaptiveCubicReg(x0, f=err, gradient=grad, hessian=hess)
        x_opt, intermediate_points, n_iter, flag = solver.adaptive_cubic_reg()
        print(n_iter)


if __name__ == '__main__':
    main()
