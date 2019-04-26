import cv2
import numpy as np
import torch

import geom_tools
import torch_rasterizer
from utils import fit_to_view_transform, transform_vertices

from timer import Timer


def rigid_transform(translation, y_rot, vertices):
    center = vertices.mean(dim=0)

    y_cos = y_rot.cos()
    y_sin = y_rot.sin()
    rot_mat = torch.stack((
        y_cos, torch.tensor(0.0), -y_sin,
        torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0),
        y_sin, torch.tensor(0.0), y_cos
    )).view(3, 3)
    res = vertices
    res = res @ rot_mat.transpose(0, 1)
    res = res - rot_mat @ center + center + translation
    return res


def render_with_shift(model, texture, canvas_size, translation, y_rot):
    torch_texture = torch.FloatTensor(texture)
    torch_texture = torch_texture.permute(2, 0, 1)

    rasterizer = torch_rasterizer.mk_rasterizer(
        model.triangle_vertex_indices,
        model.triangle_texture_vertex_indices,
        torch.FloatTensor(model.texture_vertices),
        canvas_size)

    vertices = torch.FloatTensor(model.vertices)
    vertices = rigid_transform(translation, y_rot, vertices)

    rendered = rasterizer(vertices, torch_texture)
    return rendered


def draw_uv(model, canvas_size):
    res = np.zeros(canvas_size)
    for v in model.texture_vertices:
        x = int(round(canvas_size[1] * v[0]))
        y = int(round(canvas_size[1] * v[1]))
        cv2.circle(res, (x, y), 1, 255)
    return res


def main():
    canvas_size = (256, 256)
    # canvas_size = (128, 128)
    # canvas_size = (64, 64)
    # canvas_size = (32, 32)
    # canvas_size = (16, 16)
    # path_to_obj = "/home/daiver/Girl/GirlBlendshapesWithMouthSocket/GirlWrappedNeutral.obj"

    # path_to_obj = "/home/daiver/res.obj"
    path_to_obj = "/home/daiver/res2.obj"
    path_to_texture = "/home/daiver/Girl/GirlBlendshapesWithMouthSocket/GirlNeutralFilled.jpg"

    # path_to_obj = "models/Alex1.obj"
    # path_to_texture = "models/Alex1.png"

    # path_to_obj = "models/Alex1.obj"
    # path_to_texture = "models/Alex1.png"

    model = geom_tools.from_obj_file(path_to_obj)
    mat, vec, z_min = fit_to_view_transform(model.vertices, (canvas_size[1], canvas_size[0]))
    model.vertices = transform_vertices(mat, vec, model.vertices)

    texture = cv2.imread(path_to_texture)
    texture = cv2.pyrDown(texture)
    texture = cv2.pyrDown(texture)

    # target_translation = torch.FloatTensor([5, 0, 0])
    # target_translation = torch.FloatTensor([0, 0, 0])
    target_translation = torch.FloatTensor([0, -3.75, 0])
    target_y_rotation = torch.tensor(0.7)
    torch_target_render = render_with_shift(model, texture, canvas_size, target_translation, target_y_rotation)
    cv2.imshow("target", torch_target_render.permute(1, 2, 0).detach().numpy().astype(np.uint8))
    cv2.waitKey(100)

    torch_texture = torch.FloatTensor(texture)
    torch_texture = torch_texture.permute(2, 0, 1)

    rasterizer = torch_rasterizer.mk_rasterizer(
        model.triangle_vertex_indices,
        model.triangle_texture_vertex_indices,
        torch.FloatTensor(model.texture_vertices),
        canvas_size)

    vertices_orig = torch.FloatTensor(model.vertices)
    translation = torch.FloatTensor([0, 0, 0]).requires_grad_(True)
    y_rotation = torch.tensor(0.0).requires_grad_(True)

    vertices = rigid_transform(translation, y_rotation, vertices_orig)
    rendered = rasterizer(vertices, torch_texture)
    rendered = rendered.permute(1, 2, 0).detach().numpy().astype(np.uint8)
    cv2.imshow("rendered", rendered)
    cv2.waitKey(10)

    # lr_translation = 0.001
    lr_translation = 0.0005
    # lr_rotation = 0.000001
    lr_rotation = lr_translation / 32
    for i in range(100):
        vertices = rigid_transform(translation, y_rotation, vertices_orig)
        with Timer(print_line="Rasterization elapsed: {}"):
            rendered = rasterizer(vertices, torch_texture)

        loss = (rendered - torch_target_render).pow(2).mean()
        with Timer(print_line="Backward elapsed: {}"):
            loss.backward()
        print(f"iter = {i}, loss = {loss}, ||dt|| = {translation.grad.norm()}, ||drx|| = {y_rotation.grad.norm()}")

        translation.data.sub_(lr_translation * translation.grad)
        translation.grad.zero_()

        y_rotation.data.sub_(lr_rotation * y_rotation.grad)
        y_rotation.grad.zero_()

        print(f"translation = {translation} y_rot = {y_rotation}")

        rendered = rendered.permute(1, 2, 0).detach().numpy().astype(np.uint8)
        cv2.imshow("rendered", rendered)
        cv2.waitKey(10)
    cv2.waitKey()


if __name__ == '__main__':
    main()

