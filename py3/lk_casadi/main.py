import numpy as np
import cv2

import casadi
from casadi import MX, Function


def mk_images(canvas_size):
    img1 = np.zeros(tuple(canvas_size) + (3,), dtype=np.uint8)
    img2 = np.zeros(tuple(canvas_size) + (3,), dtype=np.uint8)

    cv2.circle(img1, (64, 64), 5, (255, 255, 255), -1)
    cv2.circle(img2, (64, 70), 5, (255, 255, 255), -1)

    return img1, img2


def perform_lk(img1, img2, start_point, patch_size):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img_coords_x = np.arange(0, img1.shape[1]).astype(np.float32)
    img_coords_y = np.arange(0, img1.shape[0]).astype(np.float32)

    img1_flat = img1.ravel(order='F').astype(np.float32)
    img2_flat = img2.ravel(order='F').astype(np.float32)

    img1_interpolant = casadi.interpolant("img1_interpolant", "bspline", [img_coords_x, img_coords_y], img1_flat)
    img2_interpolant = casadi.interpolant("img1_interpolant", "bspline", [img_coords_x, img_coords_y], img2_flat)

    patch_grid = np.zeros((patch_size[0] * patch_size[1], 2), dtype=np.float32)
    counter = 0
    for x in range(patch_size[1]):
        for y in range(patch_size[0]):
            patch_grid[counter] = (x - patch_size[1] / 2, y - patch_size[0] / 2)
            counter += 1

    patch_grid = MX(patch_grid.T)

    start_point = MX(start_point).reshape((2, 1))

    img1_patch = img1_interpolant(patch_grid + start_point)

    current_point_mx = MX.sym("current_point", 2, 1)
    img2_patch = img2_interpolant(patch_grid + current_point_mx)

    diff = img1_patch - img2_patch

    variables = current_point_mx
    residuals = diff
    jacobian = casadi.jacobian(residuals, current_point_mx)

    residuals_func = Function("residuals_func", [variables], [residuals])
    jacobian_func = Function("residuals_func", [variables], [jacobian])

    n_iters = 20
    vars_values = np.array([start_point[1], start_point[0]], dtype=np.float32)
    for iter_ind in range(n_iters):
        residuals_val = residuals_func(vars_values).toarray().reshape(-1)
        jacobian_val = jacobian_func(vars_values).toarray()

        gradient_val = jacobian_val.T @ residuals_val
        gradient_norm = np.linalg.norm(gradient_val)
        loss_val = (residuals_val ** 2).sum()
        print(f"{iter_ind} / {n_iters} loss_val = {loss_val} grad_norm = {gradient_norm}")
        if gradient_norm < 1e-6:
            break

        step = np.linalg.solve(jacobian_val.T @ jacobian_val, gradient_val)
        vars_values -= step
        print(f"vars_values = {vars_values}")

    return vars_values[1], vars_values[0]


def main():
    print("casadi.__version__", casadi.__version__)
    canvas_size = (128, 128)
    img1, img2 = mk_images(canvas_size)

    res = perform_lk(img1, img2, (64, 64), patch_size=(9, 9))
    print(res)
    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)
    # cv2.waitKey()


if __name__ == '__main__':
    main()
