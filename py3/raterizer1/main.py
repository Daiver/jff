import cv2
import numpy as np
import torch

import geometry_tools

from rasterization import rasterize_barycentrics_and_z_buffer_by_triangles


# Super ineffective, i don't care
def fit_to_view_transform(vertices, width_and_height):
    width, height = width_and_height
    screen_center = (width / 2, height / 2)

    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

    x_d = x_max - x_min
    y_d = y_max - y_min
    z_d = y_max - z_min

    model_center = (x_min + x_d / 2, y_min + y_d / 2, z_min + z_d / 2)

    transformation = np.eye(4)
    transformation = np.array([
        [1, 0, 0, -model_center[0]],
        [0, 1, 0, -model_center[1]],
        [0, 0, 1, -model_center[2]],
        [0, 0, 0, 1],
    ]) @ transformation

    scale = width / x_d if width < height else height / y_d
    transformation = np.array([
        [scale, 0, 0, 0],
        [0, -scale, 0, 0],
        [0, 0, scale, 0],
        [0, 0, 0, 1],
    ]) @ transformation

    transformation = np.array([
        [1, 0, 0, screen_center[0]],
        [0, 1, 0, screen_center[1]],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]) @ transformation

    mat = transformation[:3, :3]
    vec = transformation[:3, 3]
    return mat, vec, z_min * scale


def transform_vertices(mat, vec, vertices):
    return vertices @ mat.T + vec


def main():
    canvas_size = (512, 512)
    # canvas_size = (1024, 1024)
    # canvas_size = (2048, 2048)

    barycentrics_l1l2l3 = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    barycentrics_triangle_indices = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.int32)
    barycentrics_triangle_indices[:] = -1
    z_buffer = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.float32)

    path_to_obj = "/home/daiver/Downloads/R3DS_Wrap_3.3.17_Linux/Models/Basemeshes/WrapHead.obj"
    # path_to_obj = "models/teapot.obj"
    # model = pywavefront.Wavefront(path_to_obj, collect_faces=True)
    model = geometry_tools.from_obj_file(path_to_obj)
    print("Model loaded")
    # geometry_tools = model.meshes[None]
    # print(dir(geometry_tools.materials[0]))
    # print(geometry_tools.materials[0].has_uvs)
    # print(len(geometry_tools.materials[0].vertices))
    # print(geometry_tools.materials[0].vertex_size)
    # print(geometry_tools.materials[0].vertex_format)
    #
    # print(len(model.vertices))
    #
    # vertices = np.array(model.vertices, dtype=np.float32)
    vertices = model.vertices

    mat, vec, z_min = fit_to_view_transform(vertices, (canvas_size[1], canvas_size[0]))
    vertices = transform_vertices(mat, vec, vertices)

    z_buffer[:] = z_min - abs(z_min) * 0.1

    rasterize_barycentrics_and_z_buffer_by_triangles(
        # model.meshes[None].faces,
        model.polygon_vertex_indices,
        vertices,
        barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer)

    z_buffer = (z_buffer - z_buffer.min()) / (z_buffer.max() - z_buffer.min())
    cv2.imshow("", z_buffer)
    cv2.imshow("1", barycentrics_l1l2l3)
    cv2.waitKey()


if __name__ == '__main__':
    main()
