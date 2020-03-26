import time
import os
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


def perform_lk(img1, img2, img1_point, img2_start_point, patch_size):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    down_factor = 1
    # img1 = cv2.pyrDown(img1)
    # img2 = cv2.pyrDown(img2)
    #
    # img1 = cv2.pyrDown(img1)
    # img2 = cv2.pyrDown(img2)

    img1_point = (img1_point[0] / down_factor, img1_point[1] / down_factor)
    img2_start_point = (img2_start_point[0] / down_factor, img2_start_point[1] / down_factor)

    # Compose stuff for interpolants
    # img_coords_x = np.arange(0, img1.shape[1]).astype(np.float32)
    # img_coords_y = np.arange(0, img1.shape[0]).astype(np.float32)
    img_coords_x = np.arange(0, img1.shape[0]).astype(np.float32)
    img_coords_y = np.arange(0, img1.shape[1]).astype(np.float32)

    img1_flat = img1.ravel(order='F').astype(np.float32)
    img2_flat = img2.ravel(order='F').astype(np.float32)

    inter_type = "bspline"
    inter_type = "linear"
    img1_interpolant = casadi.interpolant("img1_interpolant", inter_type, [img_coords_x, img_coords_y], img1_flat)
    img2_interpolant = casadi.interpolant("img2_interpolant", inter_type, [img_coords_x, img_coords_y], img2_flat)

    # Compose stuff for patch
    patch_grid = np.zeros((patch_size[0] * patch_size[1], 2), dtype=np.float32)
    counter = 0
    for x in range(patch_size[1]):
        for y in range(patch_size[0]):
            patch_grid[counter] = (x - patch_size[1] / 2, y - patch_size[0] / 2)
            counter += 1

    patch_grid = MX(patch_grid.T)

    # Composing residuals
    img1_point = MX([img1_point[1], img1_point[0]]).reshape((2, 1))

    img1_patch = img1_interpolant(patch_grid + img1_point)

    current_point_mx = MX.sym("current_point", 2, 1)
    img2_patch = img2_interpolant(patch_grid + current_point_mx)

    diff = img1_patch - img2_patch

    variables = current_point_mx
    residuals = diff
    time_jac_start = time.time()
    jacobian = casadi.jacobian(residuals, current_point_mx)
    print(f"jac elapsed {time.time() - time_jac_start}")

    residuals_func = Function("residuals_func", [variables], [residuals])
    jacobian_func = Function("residuals_func", [variables], [jacobian])

    # Optimization stuff
    n_iters = 200
    vars_values = np.array([img2_start_point[1], img2_start_point[0]], dtype=np.float32)
    for iter_ind in range(n_iters):
        residuals_val = residuals_func(vars_values).toarray().reshape(-1)
        jacobian_val = jacobian_func(vars_values).toarray()

        gradient_val = jacobian_val.T @ residuals_val
        loss_val = (residuals_val ** 2).sum()

        step = np.linalg.solve(jacobian_val.T @ jacobian_val, gradient_val)
        vars_values -= step
        print(f"vars_values = {vars_values}")

        gradient_norm = np.sum(np.abs(gradient_val))
        step_norm = np.sum(np.abs(step))
        error = np.linalg.norm(residuals_val, ord=1) / 255.0 / (patch_size[0] * patch_size[1])
        print(f"{iter_ind} / {n_iters} err = {error} loss_val = {loss_val} grad_norm = {gradient_norm}, step_norm={step_norm}")
        if gradient_norm < 1e-6:
            break
        if step_norm < 1e-4:
            break

    return vars_values[1] * down_factor, vars_values[0] * down_factor


def main():
    print("casadi.__version__", casadi.__version__)
    canvas_size = (128, 128)
    img1, img2 = mk_images(canvas_size)

    time_start = time.time()
    res = perform_lk(img1, img2, img1_point=(64, 64), img2_start_point=(64, 64), patch_size=(9, 9))
    # res = perform_lk(img1, img2, (64, 64), patch_size=(29, 29))
    print(f"elapsed {time.time() - time_start}")
    print(res)
    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)
    # cv2.waitKey()


if __name__ == '__main__':
    # main()
    # exit(0)

    path_to_images = "/mnt/Projects/Tracker/Front/"
    image_names = os.listdir(path_to_images)
    image_names.sort()

    first_img = cv2.imread(os.path.join(path_to_images, image_names[0]))
    print(f"first image shape {first_img.shape}")
    img1_point = (465, first_img.shape[0] - 1728)
    img2_point = img1_point

    # patch_size = (31, 31)
    # patch_size = (17, 17)
    # patch_size = (21, 21)
    patch_size = (15, 15)

    for name in image_names[70:]:
        print(f"Name = {name}")
        current_img = cv2.imread(os.path.join(path_to_images, name))

        img2_point = perform_lk(
            first_img, current_img, img1_point=img1_point, img2_start_point=img2_point, patch_size=patch_size)

        cv2.circle(current_img, (int(img2_point[0]), int(img2_point[1])), 7, (0, 255, 0), 1)
        cv2.imshow("", current_img)
        cv2.waitKey(10)
