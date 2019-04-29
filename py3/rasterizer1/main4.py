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


def blend_vertices_torch(base_vertices, blends_vertices, weights):
    assert weights.shape[0] == blends_vertices.shape[0]
    base_flat = base_vertices.reshape(-1)

    blends_flat = blends_vertices.reshape(-1, base_flat.shape[0])
    deltas_flat = blends_flat - base_flat

    res_flat = base_flat + weights @ deltas_flat
    res = res_flat.reshape(base_vertices.shape)
    return res


# TODO: remove/refactor this
def render_with_shift(model, texture, canvas_size, translation, y_rot, vertices_custom=None):
    torch_texture = torch.FloatTensor(texture)
    torch_texture = torch_texture.permute(2, 0, 1)

    rasterizer = torch_rasterizer.mk_rasterizer(
        model.triangle_vertex_indices,
        model.triangle_texture_vertex_indices,
        torch.FloatTensor(model.texture_vertices),
        canvas_size)

    vertices = model.vertices if vertices_custom is None else vertices_custom
    vertices = torch.FloatTensor(vertices)
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
    # canvas_size = (1024, 1024)
    canvas_size = (512, 512)
    # canvas_size = (256, 256)
    # canvas_size = (128, 128)
    # canvas_size = (64, 64)
    # canvas_size = (32, 32)
    # canvas_size = (16, 16)

    path_to_base = "/home/daiver/girl_base.obj"
    paths_to_blends = [
        "/home/daiver/girl_smile.obj",
        "/home/daiver/girl_scream.obj",
        "/home/daiver/girl_snarl.obj",
        "/home/daiver/girl_surprise.obj",
    ]
    path_to_texture = "/home/daiver/Girl/GirlBlendshapesWithMouthSocket/GirlNeutralFilled.jpg"

    # path_to_base = "models/Alex1.obj"
    # path_to_texture = "models/Alex1.png"

    model = geom_tools.from_obj_file(path_to_base)
    blend_geoms = [geom_tools.from_obj_file(path) for path in paths_to_blends]
    blends_vertices = np.array([g.vertices for g in blend_geoms], dtype=np.float32)

    mat, vec, z_min = fit_to_view_transform(model.vertices, (canvas_size[1], canvas_size[0]))
    model.vertices = transform_vertices(mat, vec, model.vertices)
    blends_vertices = np.array([transform_vertices(mat, vec, b) for b in blends_vertices], dtype=np.float32)

    texture = cv2.imread(path_to_texture)
    # texture = cv2.pyrDown(texture)
    # texture = cv2.pyrDown(texture)

    # target_translation = torch.FloatTensor([5, 0, 0])
    # target_translation = torch.FloatTensor([0, 0, 0])
    # target_translation = torch.FloatTensor([0, -3.75, 0])
    # target_y_rotation = torch.tensor(0.7)
    target_weights = np.array([1.0, 0, 0.0, 1.0], dtype=np.float32)
    target_vertices = blend_vertices_torch(model.vertices, blends_vertices, target_weights)
    torch_target_render = render_with_shift(model, texture, canvas_size, torch.FloatTensor([0, 0, 0]), torch.tensor(0.0), vertices_custom=target_vertices)
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
    blends_vertices_torch = torch.FloatTensor(blends_vertices)

    weights = torch.tensor([0.0, 0.0, 0.0, 0.0]).requires_grad_(True)

    vertices = blend_vertices_torch(vertices_orig, blends_vertices_torch, weights)
    rendered = rasterizer(vertices, torch_texture)
    rendered = rendered.permute(1, 2, 0).detach().numpy().astype(np.uint8)
    cv2.imshow("rendered", rendered)
    # cv2.waitKey(10)
    cv2.waitKey()

    lr = 0.0002

    for i in range(200):

        vertices = blend_vertices_torch(vertices_orig, blends_vertices_torch, weights)
        with Timer(print_line="Rasterization elapsed: {}"):
            rendered = rasterizer(vertices, torch_texture)

        loss = (rendered - torch_target_render).pow(2).mean()
        # loss = (rendered - torch_target_render).abs().mean()
        with Timer(print_line="Backward elapsed: {}"):
            loss.backward()
        print(f"iter = {i}, loss = {loss}")

        weights.data.sub_(lr * weights.grad)
        weights.grad.zero_()

        print(f"weights = {weights}")

        rendered = rendered.permute(1, 2, 0).detach().numpy().astype(np.uint8)
        cv2.imshow("rendered", rendered)
        cv2.waitKey(10)

        if i == 50:
            lr /= 4
        if i == 90:
            lr /= 2
        if i == 120:
            lr /= 4

    cv2.waitKey()


if __name__ == '__main__':
    main()

