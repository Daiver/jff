import cv2
import numpy as np
import torch

import geom_tools
import torch_rasterizer
from utils import fit_to_view_transform, transform_vertices


def main():
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

    torch_texture = torch.FloatTensor(texture)

    rasterizer = torch_rasterizer.mk_rasterizer(
        model.triangle_vertex_indices,
        model.triangle_texture_vertex_indices,
        torch.FloatTensor(model.texture_vertices),
        canvas_size)

    vertices = torch.FloatTensor(model.vertices).requires_grad_(True)

    rendered, z_buffer, _, _ = rasterizer(vertices, torch_texture)
    loss = rendered.sum()
    print(loss)
    loss.backward()

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
