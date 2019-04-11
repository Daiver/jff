import cv2
import numpy as np
import torch

from barycentric import barycoords_from_2d_triangle


def rasterize_triangle(barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer, tri_index, tri_coords_3d):
    n_rows = z_buffer.shape[0]
    n_cols = z_buffer.shape[1]
    tri_coords_xy = tri_coords_3d[:, :2]
    tri_coords_xy_int = tri_coords_xy.round().astype(np.int32)
    x_start = tri_coords_xy_int[:, 0].min()
    x_finish = tri_coords_xy_int[:, 0].max()
    y_start = tri_coords_xy_int[:, 1].min()
    y_finish = tri_coords_xy_int[:, 1].max()
    for x in range(x_start, x_finish + 1):
        if not (0 <= x < n_cols):
            continue
        for y in range(y_start, y_finish + 1):
            if not (0 <= y < n_rows):
                continue
            l1, l2, l3 = barycoords_from_2d_triangle(tri_coords_xy, (float(x), float(y)))
            is_l1_ok = 0.0 <= l1 <= 1.0
            is_l2_ok = 0.0 <= l2 <= 1.0
            is_l3_ok = 0.0 <= l3 <= 1.0
            if not (is_l1_ok and is_l2_ok and is_l3_ok):
                continue
            z_val = tri_coords_3d[0, 2] * l1 + tri_coords_3d[1, 2] * l2 + tri_coords_3d[2, 2] * l3
            if z_buffer[y, x] > z_val:
                continue
            barycentrics_l1l2l3[y, x] = [l1, l2, l3]
            barycentrics_triangle_indices[y, x] = tri_index
            z_buffer[y, x] = z_val


def main():
    canvas_size = (64, 64)
    barycentrics_l1l2l3 = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    barycentrics_triangle_indices = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.int32)
    z_buffer = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.float32)
    tri_index = 0
    tri_coords_3d = np.array([
        [1, 1, 5],
        [50, 5, 7],
        [10, 30, 3]
    ], dtype=np.float32)
    rasterize_triangle(barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer, tri_index, tri_coords_3d)

    z_buffer = (z_buffer - z_buffer.min()) / (z_buffer.max() - z_buffer.min())
    cv2.imshow("", cv2.pyrUp(z_buffer))
    cv2.waitKey()


if __name__ == '__main__':
    main()
