import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

import np_draw_tools

from data_generation import make_rectangle_dataset, make_circle_dataset, draw_rectangle_sample
from models import Encoder, Decoder
from utils import numpy_images_to_torch


def up_down_trajectory():
    res = []
    start = 0
    finish = 110
    n_steps = 200
    delta = (finish - start) / (n_steps - 1)

    start_pt = (5, 64)
    for step_ind in range(n_steps):
        step = start + delta * step_ind
        x = start_pt[0] + step
        y = start_pt[1]
        res.append((y, x))

    start_pt = (5 + finish, 64)
    for step_ind in range(n_steps):
        step = start + delta * step_ind
        x = start_pt[0] - step
        y = start_pt[1]
        res.append((y, x))

    return res


def left_right_trajectory():
    res = []
    start = 0
    finish = 110
    n_steps = 200
    delta = (finish - start) / (n_steps - 1)

    start_pt = (5, 64)
    for step_ind in range(n_steps):
        step = start + delta * step_ind
        x = start_pt[0] + step
        y = start_pt[1]
        res.append((x, y))

    start_pt = (5 + finish, 64)
    for step_ind in range(n_steps):
        step = start + delta * step_ind
        x = start_pt[0] - step
        y = start_pt[1]
        res.append((x, y))

    return res


def circle_trajectory():
    start = 0
    finish = 2 * np.pi
    n_steps = 200
    radius = 50
    center = (64, 64)

    res = []
    delta = (finish - start) / (n_steps - 1)
    for angle_ind in range(n_steps):
        angle = start + delta * angle_ind
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        res.append((x, y))

    return res


def samples_from_coords(coors, func):
    return [
        func(x) for x in coors
    ]


def main():
    n_samples_to_generate = 1500
    # circle_set = make_circle_dataset(n_samples_to_generate)
    # rect_set = make_rectangle_dataset(n_samples_to_generate)

    # coords = circle_trajectory()
    coords = left_right_trajectory()
    # coords = up_down_trajectory()
    input_set = samples_from_coords(coords, draw_rectangle_sample)

    device = "cuda"
    batch_size = 128

    encoder = Encoder().to(device)
    decoder_rect = Decoder().to(device)
    decoder_circle = Decoder().to(device)

    encoder.load_state_dict(torch.load("encoder.pt"))
    decoder_rect.load_state_dict(torch.load("decoder_rect.pt"))
    decoder_circle.load_state_dict(torch.load("decoder_circle.pt"))

    encoder.eval()
    decoder_rect.eval()
    decoder_circle.eval()

    torch_rect_set = numpy_images_to_torch(input_set)
    loader = DataLoader(torch_rect_set, batch_size=batch_size)
    index = 0
    for sample in loader:
        sample = sample.to(device)
        pred = encoder(sample)
        # pred = decoder_rect(pred)
        pred = decoder_circle(pred)
        pred = (pred.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        sample = (sample.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        for p, s in zip(pred, sample):
            to_show = [s, p]
            to_show = np_draw_tools.make_grid(to_show)
            cv2.imwrite(f"results/leftright/leftright_{index}.jpg", to_show)
            index += 1
            # cv2.imshow("", to_show)
            # cv2.waitKey(30)


if __name__ == '__main__':
    # main()
    n_samples_to_generate = 1500
    circle_set = make_circle_dataset(n_samples_to_generate)
    rect_set = make_rectangle_dataset(n_samples_to_generate)
    index = 0
    for r, c in zip(circle_set, rect_set):
        to_show = [r, c]
        to_show = np_draw_tools.make_grid(to_show)
        cv2.imwrite(f"results/samples/samples_{index}.jpg", to_show)
        index += 1
        if index > 30:
            break
