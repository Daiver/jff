import numpy as np
import scipy.sparse.linalg


def solve_least_squares(a_csc, residuals):
    grad = a_csc.T @ residuals
    h = a_csc.T @ a_csc

    np.set_printoptions(threshold=np.inf, linewidth=500)
    print(h.shape)
    print(h.toarray())
    print(np.where(~h.toarray().any(axis=1))[0])
    return -scipy.sparse.linalg.spsolve(h, grad)


def perform_gauss_newton(x0, compute_res_and_jac, n_iters=1):
    assert n_iters == 1
    x = x0.copy()
    residuals, jac = compute_res_and_jac(x)
    print(f"E(x) = { 0.5 * (residuals**2).sum() }")
    step = solve_least_squares(jac, residuals)
    x += step

    return x
