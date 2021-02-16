import cv2
import numpy as np

canvas_size = (128, 128)


def draw_rectangle_sample(center):
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    point = np.array(center).round().astype(np.int32)
    pt1 = (point[0] - 5, point[1] - 5)
    pt2 = (point[0] + 5, point[1] + 5)
    cv2.rectangle(canvas, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=-1)
    return canvas


def draw_circle_sample(center):
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    point = np.array(center).round().astype(np.int32)
    cv2.circle(canvas, (point[0], point[1]), 5, (0, 255, 0), thickness=-1)
    return canvas


def draw_cross_sample(center):
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    point = np.array(center).round().astype(np.int32)

    pt1 = (point[0] - 5, point[1] - 5)
    pt2 = (point[0] + 5, point[1] + 5)
    cv2.line(canvas, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=2)

    pt3 = (point[0] + 5, point[1] - 5)
    pt4 = (point[0] - 5, point[1] + 5)
    cv2.line(canvas, pt1=pt3, pt2=pt4, color=(0, 255, 0), thickness=2)

    return canvas


def make_rectangle_dataset(n_samples_to_generate):
    res = []
    for i in range(n_samples_to_generate):
        x, y = np.random.uniform(0, canvas_size[0]), np.random.uniform(0, canvas_size[0])
        res.append(draw_rectangle_sample((x, y)))
    return res


def make_circle_dataset(n_samples_to_generate):
    res = []
    for i in range(n_samples_to_generate):
        x, y = np.random.uniform(0, canvas_size[0]), np.random.uniform(0, canvas_size[0])
        # res.append(draw_circle_sample((x, y)))
        res.append(draw_cross_sample((x, y)))
    return res


