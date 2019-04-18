import cv2
import numpy as np
import torch

import geom_tools
import torch_rasterizer
from utils import fit_to_view_transform, transform_vertices


def render_with_shift(model, texture, canvas_size, shift):
    torch_texture = torch.FloatTensor(texture)

    rasterizer = torch_rasterizer.mk_rasterizer(
        model.triangle_vertex_indices,
        model.triangle_texture_vertex_indices,
        torch.FloatTensor(model.texture_vertices),
        canvas_size)

    vertices = torch.FloatTensor(model.vertices)
    vertices = vertices + shift

    rendered, z_buffer, _, _ = rasterizer(vertices, torch_texture)
    return rendered


def main():
    # canvas_size = (256, 256)
    # canvas_size = (128, 128)
    canvas_size = (64, 64)
    path_to_obj = "/home/daiver/Girl/GirlBlendshapesWithMouthSocket/GirlWrappedNeutral.obj"
    path_to_texture = "/home/daiver/Girl/GirlBlendshapesWithMouthSocket/GirlNeutralFilled.jpg"

    # path_to_obj = "models/Alex1.obj"
    # path_to_texture = "models/Alex1.png"

    model = geom_tools.from_obj_file(path_to_obj)
    mat, vec, z_min = fit_to_view_transform(model.vertices, (canvas_size[1], canvas_size[0]))
    model.vertices = transform_vertices(mat, vec, model.vertices)

    texture = cv2.imread(path_to_texture)
    texture = cv2.pyrDown(texture)
    texture = cv2.pyrDown(texture)

    # target_shift = torch.FloatTensor([5, 0, 0])
    target_shift = torch.FloatTensor([0, 0, 0])
    torch_target_render = render_with_shift(model, texture, canvas_size, target_shift)
    cv2.imshow("target", torch_target_render.permute(1, 2, 0).detach().numpy() / 255)

    torch_texture = torch.FloatTensor(texture)

    rasterizer = torch_rasterizer.mk_rasterizer(
        model.triangle_vertex_indices,
        model.triangle_texture_vertex_indices,
        torch.FloatTensor(model.texture_vertices),
        canvas_size)

    vertices = torch.FloatTensor(model.vertices).requires_grad_(True)
    vertices = vertices

    rendered, z_buffer, _, _ = rasterizer(vertices, torch_texture)
    loss = (rendered - torch_target_render).pow(2).sum()
    print(loss)
    loss.backward()
    print(vertices.grad)

    rendered = rendered.permute(1, 2, 0).detach().numpy() / 255
    print(rendered.shape)

    z_buffer = z_buffer.detach().numpy()

    z_diff = (z_buffer.max() - z_buffer.min())
    print(z_diff)
    z_buffer = (z_buffer - z_buffer.min()) / z_diff
    cv2.imshow("", z_buffer)
    cv2.imshow("rendered", rendered)
    cv2.waitKey()


if __name__ == '__main__':
    main()
