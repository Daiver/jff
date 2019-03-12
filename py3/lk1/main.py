import os
import torch
import scipy
import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

# window_size = (11, 11)
# window_size = (41, 41)
# window_size = (21, 21)
# window_size = (61, 61)
window_size = (41, 41)


def draw_circle(canvas, point, radius=5):
    cv2.circle(canvas, (int(round(point[0])), int(round(point[1]))), radius, color=255, thickness=-1)


def im_grad_x(img):
    img = img.astype(np.float32)
    kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel=kernel)


def im_grad_y(img):
    img = img.astype(np.float32)
    kernel = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel=kernel)


def cut_patch(img, patch_center):
    coords = []
    for row in range(- window_size[1] // 2, window_size[1] // 2):
        for col in range(- window_size[0] // 2, window_size[0] // 2):
            coords.append([row + patch_center[1], col + patch_center[0]])

    coords = np.array(coords)
    res = map_coordinates(img, coords.T, order=1)
    return res


def compute_jacobian(frame0_x, frame0_y, patch_center):
    dxs = cut_patch(frame0_x, patch_center)
    dys = cut_patch(frame0_y, patch_center)
    return np.stack((dxs, dys), axis=1)


def compute_lk_error(frame0, frame1, patch_center, p):
    patch_center = np.array(patch_center, dtype=np.float32)
    p = np.array(p, dtype=np.float32)
    vals0 = cut_patch(frame0, patch_center)
    vals1 = cut_patch(frame1, patch_center + p)
    diff = vals0 - vals1
    return diff.T @ diff


def perform_lk(frame0, frame1, patch_center, p0):
    p = np.array([p0[0], p0[1]], dtype=np.float32)
    print(p)

    frame0_x = im_grad_x(frame0)
    frame0_y = im_grad_y(frame0)

    jacobian = compute_jacobian(frame0_x, frame0_y, patch_center)
    hessian = jacobian.T @ jacobian
    hessian_inv = np.linalg.inv(hessian)

    n_iters = 200
    for iter_ind in range(n_iters):
        err = compute_lk_error(frame0, frame1, patch_center, p)
        print(f"{iter_ind + 1}/{n_iters} Loss = {err}")
        if err < 1e-3:
            break

        vals0 = cut_patch(frame0, patch_center)
        vals1 = cut_patch(frame1, patch_center + p)
        grad = jacobian.T @ (vals0 - vals1)
        dp = hessian_inv @ grad
        p += dp
        print(p)
        if np.linalg.norm(dp) < 1e-4:
            break
    return p


def main1():
    canvas1 = np.zeros((128, 128), dtype=np.uint8)
    canvas2 = np.zeros((128, 128), dtype=np.uint8)

    draw_circle(canvas1, (64, 64))
    draw_circle(canvas2, (66, 64))

    print(perform_lk(canvas1.astype(np.float32), canvas2.astype(np.float32), (64, 64), (5, -9)))


def main2():
    frame0 = cv2.imread("/home/daiver/1_ed.tiff", 0)
    frame1 = cv2.imread("/home/daiver/2_ed.tiff", 0)

    patch_center = (52, 60)
    res = perform_lk(frame0.astype(np.float32), frame1.astype(np.float32), patch_center, (0, 0))
    print(res)
    cv2.circle(frame0, patch_center, 15, 255)
    cv2.circle(frame1, (int(round(patch_center[0] + res[0])), int(round(patch_center[1] + res[1]))), 15, 255)

    cv2.imshow("1", frame0)
    cv2.imshow("2", frame1)
    cv2.waitKey()


def draw_marker_and_window(img, patch_center):
    radius = window_size[0] // 2
    cv2.circle(img, (int(round(patch_center[0])), int(round(patch_center[1]))), 2, 255, -1)
    cv2.circle(img, (int(round(patch_center[0])), int(round(patch_center[1]))), radius, 255)


if __name__ == '__main__':
    print("Hi")

    # patch_center = (573, 333)
    # patch_center = (800, 887)
    # patch_center = (457, 1079)
    # patch_center = (420, 632)
    patch_center = (419, 581)
    patch_center = np.array(patch_center, dtype=np.float32) / 2.0

    data_root = "/work/R3DS/Data/SashaFrontView/"
    names = [os.path.join(data_root, x) for x in os.listdir(data_root)]
    names.sort()
    res_dir = "/home/daiver/res7"
    os.makedirs(res_dir, exist_ok=True)
    for i in range(1, len(names)):
        name0 = names[i]
        name1 = names[i + 1]
        print(name0, name1)
        frame0 = cv2.imread(name0, 0)
        frame1 = cv2.imread(name1, 0)
        frame0 = cv2.pyrDown(frame0)
        frame1 = cv2.pyrDown(frame1)

        res = perform_lk(frame0.astype(np.float32), frame1.astype(np.float32), patch_center, (0, 0))

        radius = window_size[0] // 2
        draw_marker_and_window(frame0, patch_center)
        draw_marker_and_window(frame1, patch_center + res)
        # cv2.circle(frame0, (int(round(patch_center[0])), int(round(patch_center[1]))), radius, 255)
        # cv2.circle(frame1, (int(round(patch_center[0] + res[0])), int(round(patch_center[1] + res[1]))), radius, 255)
        patch_center = (patch_center[0] + res[0], patch_center[1] + res[1])

        if i == 1:
            cv2.imwrite(os.path.join(res_dir, f"0.png"), frame0)
        cv2.imwrite(os.path.join(res_dir, f"{i}.png"), frame1)

        # cv2.imshow("0", frame0)
        # cv2.imshow("1", frame1)
        # cv2.waitKey()


