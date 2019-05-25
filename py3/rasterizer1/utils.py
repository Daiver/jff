import numpy as np


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

    scale = min(width / x_d, height / y_d)
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
    return mat, vec, (z_min - model_center[2]) * scale


def transform_vertices(mat, vec, vertices):
    return vertices @ mat.T + vec
