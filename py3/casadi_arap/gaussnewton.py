import numpy as np
import scipy.sparse.linalg


def solve_least_squares(a_csc, residuals, dumping_factor):
    grad = a_csc.T @ residuals
    print("Before h")
    h = a_csc.T @ a_csc

    np.set_printoptions(threshold=np.inf, linewidth=500)
    # print(h.shape)
    # print(h.toarray())
    # print(np.where(~h.toarray().any(axis=1))[0])
    h += dumping_factor * scipy.sparse.eye(h.shape[0])
    print("Before solve")
    return -scipy.sparse.linalg.spsolve(h, grad)


def perform_gauss_newton(x0, compute_res_and_jac, n_iters=10, dumping_factor=0.0):
    threshold_err = 1e-20

    x = x0.copy()
    residuals, jac = compute_res_and_jac(x)
    grad = jac.T @ residuals
    grad_norm = np.linalg.norm(grad)
    err = 0.5 * (residuals ** 2).sum()

    print(f"E(x0) = {err}, ||∇E(x0)||_2 = {grad_norm}")
    for iter_num in range(n_iters):
        step = solve_least_squares(jac, residuals, dumping_factor)
        print("After solve")
        x += step

        residuals, jac = compute_res_and_jac(x)
        grad = jac.T @ residuals
        grad_norm = np.linalg.norm(grad)
        err = 0.5 * (residuals ** 2).sum()
        print(f"{iter_num + 1} / {n_iters} E(x) = {err}, ||∇E(x)||_2 = {grad_norm}")

        if err < threshold_err:
            break

    return x
