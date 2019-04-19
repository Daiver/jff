import torch
import torch.autograd
# import torch.nn as nn
import torch.nn.functional as F

from barycentric import barycoords_from_2d_triangle
import torch_img_gradient


def rasterize_triangle(barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer, tri_index, tri_coords_3d):
    n_rows = z_buffer.shape[0]
    n_cols = z_buffer.shape[1]
    tri_coords_xy = tri_coords_3d[:, :2]
    tri_coords_xy_int = tri_coords_xy.round().int()
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
            is_l1_ok = 0.0 - 1e-7 <= l1 <= 1.0 + 1e-7
            is_l2_ok = 0.0 - 1e-7 <= l2 <= 1.0 + 1e-7
            is_l3_ok = 0.0 - 1e-7 <= l3 <= 1.0 + 1e-7
            if not (is_l1_ok and is_l2_ok and is_l3_ok):
                continue
            z_val = tri_coords_3d[0, 2] * l1 + tri_coords_3d[1, 2] * l2 + tri_coords_3d[2, 2] * l3
            if z_buffer[y, x] > z_val:
                continue
            barycentrics_l1l2l3[y, x, 0] = l1
            barycentrics_l1l2l3[y, x, 1] = l2
            barycentrics_l1l2l3[y, x, 2] = l3
            barycentrics_triangle_indices[y, x] = tri_index
            z_buffer[y, x] = z_val


def rasterize_barycentrics_and_z_buffer_by_triangles(
        triangle_vertex_indices, vertices,
        barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer):
    for tri_index, face in enumerate(triangle_vertex_indices):

        tri_coords_3d = torch.stack((
            vertices[face[0]],
            vertices[face[1]],
            vertices[face[2]],
        ))
        rasterize_triangle(barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer, tri_index, tri_coords_3d)


def grid_for_texture_warp(
        barycentrics_l1l2l3,
        barycentrics_triangle_indices,
        texture_vertices,
        triangle_texture_vertex_indices):
    res = torch.zeros((barycentrics_l1l2l3.shape[0], barycentrics_l1l2l3.shape[1], 2))

    for x in range(res.shape[1]):
        for y in range(res.shape[0]):
            tri_ind = barycentrics_triangle_indices[y, x]
            if tri_ind == -1:
                continue
            l1, l2, l3 = barycentrics_l1l2l3[y, x]
            i1, i2, i3 = triangle_texture_vertex_indices[tri_ind]
            vt1, vt2, vt3 = texture_vertices[i1], texture_vertices[i2], texture_vertices[i3]
            final_coord = vt1 * l1 + vt2 * l2 + vt3 * l3
            res[y, x] = final_coord

    return res


def normalize_grid_for_grid_sample(torch_grid):
    torch_grid = torch.stack((torch_grid[:, :, 0], 1.0 - torch_grid[:, :, 1]), dim=2)
    torch_grid = torch_grid * 2 - 1
    return torch_grid


def warp_grid_torch(torch_mask, torch_grid, torch_texture):
    """

    :param torch_mask:
    :param torch_grid:
    :param torch_texture: CxHxW tensor
    :return:
    """
    assert len(torch_texture.shape) == 3
    torch_texture = torch_texture.unsqueeze(0)

    torch_grid = normalize_grid_for_grid_sample(torch_grid)

    torch_grid = torch_grid.unsqueeze(0)
    res = F.grid_sample(torch_texture, torch_grid, mode="bilinear").squeeze(0)

    res = res * torch_mask
    return res


def vertices_grad(
        inp_grad,
        torch_warped_dx, torch_warped_dy,
        triangle_vertex_indices,
        barycentrics_l1l2l3, barycentrics_triangle_indices,
        n_vertices
):
    # grad with respect to z direction is always zero
    res = torch.zeros((n_vertices, 3))
    assert torch_warped_dx.shape == inp_grad.shape
    n_channels, n_rows, n_cols = torch_warped_dx.shape

    inp_grad = inp_grad.permute(1, 2, 0)  # c, h, w -> h, w, c
    torch_warped_dx = torch_warped_dx.permute(1, 2, 0)
    torch_warped_dy = torch_warped_dy.permute(1, 2, 0)

    for row in range(n_rows):
        for col in range(n_cols):
            tri_ind = barycentrics_triangle_indices[row, col]
            if tri_ind == -1:
                continue

            for v_index, l in zip(triangle_vertex_indices[tri_ind], barycentrics_l1l2l3[row, col]):
                res[v_index, 0] += l * torch_warped_dx[row, col].dot(inp_grad[row, col])
                res[v_index, 1] += l * torch_warped_dy[row, col].dot(inp_grad[row, col])

    return res


def mk_rasterizer(
        triangle_vertex_indices,
        triangle_texture_vertex_indices,
        texture_vertices,
        canvas_size,
        return_z_buffer=False,
        return_barycentrics=False
):

    class Rasterizer(torch.autograd.Function):
        def __init__(self):
            super().__init__()

        @staticmethod
        def forward(ctx, vertices, texture):
            barycentrics_l1l2l3 = torch.zeros((canvas_size[0], canvas_size[1], 3))
            barycentrics_triangle_indices = torch.zeros((canvas_size[0], canvas_size[1])).int()
            barycentrics_triangle_indices[:] = -1
            z_buffer = torch.zeros((canvas_size[0], canvas_size[1]))

            z_min = vertices[:, 2].min()
            z_buffer[:] = z_min  # - 1e-3 * abs(z_min)

            rasterize_barycentrics_and_z_buffer_by_triangles(
                triangle_vertex_indices,
                vertices,
                barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer)

            torch_mask = (barycentrics_triangle_indices != -1).float()

            torch_grid = grid_for_texture_warp(
                barycentrics_l1l2l3, barycentrics_triangle_indices,
                texture_vertices, triangle_texture_vertex_indices)

            torch_warped = warp_grid_torch(torch_mask, torch_grid, texture)
            torch_warped_dx = torch_img_gradient.img_grad_dx(torch_warped)
            torch_warped_dy = torch_img_gradient.img_grad_dy(torch_warped)

            ctx.mark_non_differentiable(z_buffer, barycentrics_l1l2l3, barycentrics_triangle_indices, texture)
            ctx.save_for_backward(
                torch_warped, torch_warped_dx, torch_warped_dy,
                barycentrics_l1l2l3, barycentrics_triangle_indices)

            if not return_z_buffer and not return_barycentrics:
                return torch_warped
            res_list = [torch_warped]
            if return_z_buffer:
                res_list.append(z_buffer)
            if return_barycentrics:
                res_list.append(barycentrics_l1l2l3)
                res_list.append(barycentrics_triangle_indices)
            return tuple(res_list)

        @staticmethod
        def backward(ctx, *grad_outputs):
            torch_warped, torch_warped_dx, torch_warped_dy, barycentrics_l1l2l3, barycentrics_triangle_indices = \
                ctx.saved_tensors
            inp_render_grad = grad_outputs[0]

            # TODO: MAKE IT EXPLICIT!
            n_vertices = 1 + max(max(x) for x in triangle_vertex_indices)

            vertices_res_grad = vertices_grad(
                inp_render_grad,
                torch_warped_dx, torch_warped_dy,
                triangle_vertex_indices, barycentrics_l1l2l3, barycentrics_triangle_indices,
                n_vertices
            )
            return vertices_res_grad, None

    return Rasterizer.apply
