import numpy as np
import cv2

import casadi
from casadi import SX, MX, DM


def mk_images(canvas_size):
    img1 = np.zeros(tuple(canvas_size) + (3,), dtype=np.uint8)
    img2 = np.zeros(tuple(canvas_size) + (3,), dtype=np.uint8)

    cv2.circle(img1, (64, 64), 5, (255, 255, 255), -1)
    cv2.circle(img2, (64, 70), 5, (255, 255, 255), -1)

    return img1, img2


def perform_lk(img1, img2, start_point, patch_size=(9, 9)):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img_coords_x = np.arange(0, img1.shape[1]).astype(np.float32)
    img_coords_y = np.arange(0, img1.shape[0]).astype(np.float32)

    img1_flat = img1.ravel(order='F').astype(np.float32)
    img2_flat = img2.ravel(order='F').astype(np.float32)

    img1_interpolant = casadi.interpolant("img1_interpolant", "bspline", [img_coords_x, img_coords_y], img1_flat)
    img2_interpolant = casadi.interpolant("img1_interpolant", "bspline", [img_coords_x, img_coords_y], img2_flat)
    #
    # print(img1_interpolant(DM(np.array([[0, 0], [64, 64]]).T)))
    # print(img2_interpolant(DM(np.array([[0, 0], [70, 64]]).T)))

    patch_grid = np.zeros((patch_size[0] * patch_size[1], 2), dtype=np.float32)
    counter = 0
    for x in range(patch_size[1]):
        for y in range(patch_size[0]):
            patch_grid[counter] = (x, y)
            counter += 1

    patch_grid = DM(patch_grid.T)
    print(img1_interpolant(patch_grid))

    point_mx = MX.sym("point", 2, 1)
    #
    # xgrid = np.linspace(-5,5,11)
    # ygrid = np.linspace(-4,4,9)
    # X,Y = np.meshgrid(xgrid,ygrid,indexing='ij')
    # R = np.sqrt(5*X**2 + Y**2)+ 1
    # data = np.sin(R)/R
    # data_flat = data.ravel(order='F')
    # lut = casadi.interpolant('name','bspline',[xgrid,ygrid],data_flat)
    # print(lut([0.5,1]))


def main():
    canvas_size = (128, 128)
    img1, img2 = mk_images(canvas_size)

    perform_lk(img1, img2, (64, 64))
    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)
    # cv2.waitKey()


if __name__ == '__main__':
    main()
