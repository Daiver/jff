import torch
import torch.autograd

from barycentric import barycoords_from_2d_triangle


class Rasterizer(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    def forward(self, ctx, vertices, texture):
        pass

    def backward(self, ctx, *grad_outputs):
        assert False


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