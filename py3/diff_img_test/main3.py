import cv2

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from losses import mk_diff_img


def sample_lines(lines, n_points):
    assert n_points >= 2
    lines = np.array(lines)
    assert lines.shape[1] == 2
    res = []
    space = np.linspace(0, 1, n_points)

    for line in lines:
        diff = line[1] - line[0]
        res += (space.reshape(n_points, 1) @ diff.reshape(1, 2) + line[0]).tolist()
    return np.array(res)


def gen_lines(offset=(0, 0), scale=1.0):
    lines = [
        [(-1, 0), (1, 0)],
    ]
    return np.array(lines) * scale + offset


def gen_blendshape_lines(scale=1.0):
    shapes = [
        [
            [(-1, 0), (1, 0)]
        ],
    ]
    return np.array(shapes) * scale


def draw_lines(lines, img, color, thickness=1):
    lines = np.array(lines).round().astype(np.int32)
    for line in lines:
        cv2.line(img, (line[0, 1], line[0, 0]), (line[1, 1], line[1, 0]), color=color, thickness=thickness)


def draw_points(points, img, color, radius=2):
    points = np.array(points).round().astype(np.int32)
    for point in points:
        cv2.circle(img, (point[1], point[0]), radius=radius, color=color, thickness=-1)


image_size = (64, 64)
default_line_offset = np.array((32, 32))
default_line_scale = 12
sampling_rate = 5


def gen_random_example(image_size):
    rand_translate = np.random.normal(scale=4, size=2)
    # print('rand_translate =', rand_translate)
    line_offset = default_line_offset + rand_translate
    line_scale = default_line_scale
    neutral_lines = gen_lines(line_offset, line_scale)

    target_img = np.zeros(image_size, dtype=np.float32)
    draw_lines(neutral_lines, target_img, 1, 1)
    target_img = cv2.GaussianBlur(target_img, (15, 15), 1.0)
    return target_img, neutral_lines, rand_translate


def main():

    neutral_lines = gen_lines(default_line_offset, default_line_scale)
    blendshape_lines = gen_blendshape_lines(default_line_scale)
    sampled_neutral = sample_lines(neutral_lines, sampling_rate)
    sampled_blendshapes = [sample_lines(lines, sampling_rate) for lines in blendshape_lines]
    print(sampled_blendshapes)

    target_img, target_shape, target_translate = gen_random_example(image_size)
    cv2.imshow('', cv2.pyrUp(cv2.pyrUp(target_img)))
    cv2.waitKey(10)

    target_img = torch.FloatTensor(target_img)
    target_img = target_img.cuda()
    diff_img = mk_diff_img(target_img)

    target_colors = torch.FloatTensor(np.ones(len(sampled_neutral))).cuda()

    n_blends = len(blendshape_lines)
    weights = torch.FloatTensor([0]).cuda()
    assert len(weights) == n_blends
    translation = torch.FloatTensor([0, 0]).cuda()

    weights.requires_grad_()
    translation.requires_grad_()

    sampled_neutral = torch.FloatTensor(sampled_neutral).cuda()
    sampled_blendshapes = torch.FloatTensor(sampled_blendshapes).cuda()
    print(sampled_blendshapes)

    epochs = 100
    lr = 0.07
    optimizer = optim.Adam([translation, weights], lr=lr)
    for iter_ind in range(epochs):
        # blended = (sampled_blendshapes.view(-1) * weights[0]).view(-1, 2)
        points = sampled_neutral + translation
        pixels = diff_img.apply(points)
        diff = pixels - target_colors

        loss = diff.norm(2) / len(points)
        print('loss =', loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f'translation = {translation}')
    # print(f'weights = {weights}')
    print(f'translation target = {target_translate}')

    canvas = cv2.cvtColor(target_img.detach().cpu().numpy(), cv2.COLOR_GRAY2BGR)
    draw_points(points.detach().cpu().numpy(), canvas, (0, 255, 0), 1)
    cv2.imshow('canvas', cv2.pyrUp(cv2.pyrUp(canvas)))

    # print('target_shape =', target_shape)
    # print('result shape =', sampled_neutral + translation)
    # print(weights.data)
    # sampled_neutral = sampled_neutral - 0.5 * sampled_blendshapes[0]

    # canvas = np.zeros(image_size + (3,), dtype=np.uint8)
    # draw_lines(neutral_lines, canvas, (255, 255, 255))
    # draw_points(sampled_neutral, canvas, (0, 255, 0), 2)
    # cv2.imshow('1', cv2.pyrUp(cv2.pyrUp(canvas)))
    cv2.waitKey()


if __name__ == '__main__':
    main()
