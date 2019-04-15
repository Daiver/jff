import cv2
import numpy as np
import torch

import geom_tools

from utils import fit_to_view_transform, transform_vertices
from rasterization import (
    rasterize_barycentrics_and_z_buffer_by_triangles, grid_for_texture_warp, warp_grid_numpy, warp_grid_torch)


def main():
    # path_to_obj = "/home/daiver/Downloads/R3DS_Wrap_3.3.17_Linux/Models/Basemeshes/WrapHead.obj"
    # path_to_texture = "/home/daiver/chess.jpg"

    # path_to_obj = "/home/daiver/Girl/GirlBlendshapesWithMouthSocket/GirlWrappedNeutral.obj"
    # path_to_texture = "/home/daiver/Girl/GirlBlendshapesWithMouthSocket/GirlNeutralFilled.jpg"

    path_to_obj = "models/Alex1.obj"
    path_to_texture = "models/Alex1.png"

    model = geom_tools.from_obj_file(path_to_obj)

    img = cv2.imread(path_to_texture)

    target = cv2.imread("render1.png")

    # canvas_size = (64, 64)
    canvas_size = (256, 256)
    # canvas_size = (512, 512)
    # canvas_size = (1024, 1024)
    # canvas_size = (2048, 2048)

    barycentrics_l1l2l3 = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    barycentrics_triangle_indices = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.int32)
    barycentrics_triangle_indices[:] = -1
    z_buffer = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.float32)

    vertices = model.vertices

    mat, vec, z_min = fit_to_view_transform(vertices, (canvas_size[1], canvas_size[0]))
    print(z_min)
    vertices = transform_vertices(mat, vec, vertices)

    z_buffer[:] = z_min - abs(z_min) * 0.1

    rasterize_barycentrics_and_z_buffer_by_triangles(
        model.triangle_vertex_indices,
        vertices,
        barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer)
    grid = grid_for_texture_warp(
        barycentrics_l1l2l3, barycentrics_triangle_indices,
        model.texture_vertices, model.triangle_texture_vertex_indices)

    torch_image = torch.FloatTensor(img).float()
    torch_image.requires_grad_(True)
    torch_warped = warp_grid_torch(barycentrics_triangle_indices, grid, torch_image)
    torch_target = torch.FloatTensor(target).transpose(2, 0)
    print(torch_warped.requires_grad, torch_warped.shape, torch_target.shape)
    loss = (torch_warped - torch_target).pow(2).sum()
    print(loss)
    loss.backward()
    warped = (torch_warped.transpose(0, 2) / 255).detach().numpy()

    print(warped.shape, warped.dtype)
    warped = (warped * 255).astype(np.uint8)
    cv2.imshow("warped", warped)

    target2 = torch_target.transpose(2, 0).detach().numpy().astype(np.uint8)
    print("err", ((target2 - warped) ** 2).sum())
    cv2.imshow("t2", target2)

    z_buffer_diff = z_buffer.max() - z_buffer.min()
    if z_buffer_diff < 1e-6:
        z_buffer_diff = 1
    z_buffer = (z_buffer - z_buffer.min()) / z_buffer_diff
    z_buffer = (255 * z_buffer).astype(np.uint8)
    cv2.imshow("", z_buffer)
    cv2.waitKey()


if __name__ == '__main__':
    main()
