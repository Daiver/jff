import time

import cv2
import numpy as np

import utils


def generate_dataset():
    n_x_steps = 85
    # n_x_steps = 2
    n_y_steps = 1

    point_orig = [25, 50]
    images = []
    positions = []

    for dx in range(n_x_steps):
        point = [point_orig[0] + dx, point_orig[1]]
        positions.append(point)
        images.append(draw_sample(point))

    for dx in range(n_x_steps):
        point = [point_orig[0] + n_x_steps - dx, point_orig[1] + 5]
        positions.append(point)
        images.append(draw_sample(point))

    return images, positions


def generate_dataset_and_compute_flow():
    images, positions = generate_dataset()

    time_start = time.time()
    flows = utils.compute_flow_for_clip(images)
    print(f"elapsed {time.time() - time_start}")
    return images, positions, flows


def draw_sample(coord):
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
    x, y = np.array(coord).round().astype(np.int32)
    cv2.circle(canvas, center=(x, y), radius=10, color=(0, 255, 0), thickness=-1)
    return canvas


canvas_size = (128, 128)