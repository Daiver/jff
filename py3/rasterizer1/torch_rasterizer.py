import torch
import torch.autograd
import torch.nn.functional as F

from barycentric import barycoords_from_2d_triangle


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
    torch_grid = torch_grid.transpose(0, 1)

    torch_texture = torch_texture.transpose(2, 0)
    torch_texture = torch_texture.transpose(1, 2)
    torch_texture = torch_texture.unsqueeze(0)

    torch_grid = normalize_grid_for_grid_sample(torch_grid)

    torch_grid = torch_grid.unsqueeze(0)
    res = F.grid_sample(torch_texture, torch_grid, mode="bilinear").squeeze()

    res = res * torch_mask
    return res


def mk_rasterizer(
    triangle_vertex_indices,
    triangle_texture_vertex_indices,
    canvas_size):

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

            ctx.mark_non_differentiable(z_buffer, barycentrics_l1l2l3, barycentrics_triangle_indices)
            return z_buffer.clone(), z_buffer, barycentrics_l1l2l3, barycentrics_triangle_indices

        @staticmethod
        def backward(ctx, *grad_outputs):
            assert False

    return Rasterizer.apply


"""
    barycentrics_l1l2l3 = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    barycentrics_triangle_indices = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.int32)
    barycentrics_triangle_indices[:] = -1
    z_buffer = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.float32)    

    z_min = vertices[:, 2].min()
    z_buffer[:] = z_min

    rasterize_barycentrics_and_z_buffer_by_triangles(
        model.triangle_vertex_indices,
        vertices,
        barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer)
        
    torch_mask = torch.FloatTensor((barycentrics_triangle_indices != -1).astype(np.float32))
    torch_mask = torch_mask.transpose(0, 1)
        
    torch_grid = grid_for_texture_warp(
        barycentrics_l1l2l3, barycentrics_triangle_indices,
        model.texture_vertices, model.triangle_texture_vertex_indices)    
        
    torch_warped = warp_grid_torch(torch_mask, torch_grid, torch_texture)        
"""