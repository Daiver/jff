import cv2
import numpy as np
import torch

import geom_tools
import torch_rasterizer
from utils import fit_to_view_transform, transform_vertices

from timer import Timer


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
    # canvas_size = (16, 16)
    # canvas_size = (32, 32)
    # path_to_obj = "/home/daiver/Girl/GirlBlendshapesWithMouthSocket/GirlWrappedNeutral.obj"

    # path_to_obj = "/home/daiver/res.obj"
    path_to_obj = "/home/daiver/res2.obj"
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
    target_shift = torch.FloatTensor([-6, -4, 0])
    # target_shift = torch.FloatTensor([0, 0, 0])
    torch_target_render = render_with_shift(model, texture, canvas_size, target_shift)
    cv2.imshow("target", torch_target_render.permute(1, 2, 0).detach().numpy() / 255)

    torch_texture = torch.FloatTensor(texture)

    rasterizer = torch_rasterizer.mk_rasterizer(
        model.triangle_vertex_indices,
        model.triangle_texture_vertex_indices,
        torch.FloatTensor(model.texture_vertices),
        canvas_size)

    vertices_orig = torch.FloatTensor(model.vertices)
    translation = torch.FloatTensor([0, 0, 0]).requires_grad_(True)

    lr = 0.0005
    for i in range(100):
        vertices = vertices_orig + translation
        with Timer(print_line="Rasterization elapsed: {}"):
            rendered, z_buffer, _, _ = rasterizer(vertices, torch_texture)
        loss = (rendered - torch_target_render).pow(2).mean()
        print(i, loss)
        with Timer(print_line="Backward elapsed: {}"):
            loss.backward()
        # print(translation.grad.max())
        # translation.data.sub_(lr * translation.grad)
        translation.data.add_(lr * translation.grad)
        translation.grad.zero_()
        print(f"translation = {translation}")

        rendered = rendered.permute(1, 2, 0).detach().numpy() / 255
        cv2.imshow("rendered", rendered)
        cv2.waitKey(10)
    cv2.waitKey()


if __name__ == '__main__':
    main()
