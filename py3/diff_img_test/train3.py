import random
import itertools
import time
import cv2

import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from losses import mk_diff_img, points_render_loss
from models import Model


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
        [(-1, 0), (0, 1)],
        [(0, 1), (1, 0)],
        [(1, 0), (0, -1)],
        [(0, -1), (-1, 0)],
    ]
    return np.array(lines) * scale + offset


def gen_blendshape_lines(scale=1.0):
    shapes = [
        [
            [(-1, 0), (0, 1)],
            [(0, 1), (1, 0)],
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
    # rand_translate = np.random.normal(scale=8, size=2)
    rand_translate = np.random.uniform(-10, 10, size=2)
    # print('rand_translate =', rand_translate)
    line_offset = default_line_offset + rand_translate
    line_scale = default_line_scale
    neutral_lines = gen_lines(line_offset, line_scale)

    target_img = np.zeros(image_size, dtype=np.float32)
    draw_lines(neutral_lines, target_img, 1, 1)
    # target_img = cv2.GaussianBlur(target_img, (15, 15), 0.5)
    return target_img, neutral_lines, rand_translate


def main2():
    n_targets = 3000

    target_images = []
    target_translations = []
    for i in range(n_targets):
        target_img, _, target_translate = gen_random_example(image_size)
        target_images.append(target_img)
        target_translations.append(target_translate)

        del target_img
        del target_translate

    for i, img in enumerate(target_images):
        cv2.imwrite(f'tmp/{i}.png', (img * 255).astype(np.uint8))

    neutral_lines = gen_lines(default_line_offset, default_line_scale)
    sampled_neutral = sample_lines(neutral_lines, sampling_rate)
    sampled_neutral = torch.FloatTensor(sampled_neutral)

    sampled_neutral = sampled_neutral.cuda()

    model = Model()
    model = model.cuda()

    epochs = 1000
    # lr = 0.002
    # lr = 0.0002
    # lr = 0.00005
    lr = 0.02
    batch_size = 64
    optimizer = optim.Adam(model.parameters(), lr=lr)

    target_images2 = [torch.FloatTensor(x) for x in target_images]
    loader = DataLoader(target_images2, batch_size=batch_size, shuffle=True)

    for iter_ind in range(epochs):
        losses = []
        norms = []
        start_time = time.time()
        model.train()
        for target_img_batch in loader:
            target_img_batch = target_img_batch.cuda()

            translations = model(target_img_batch.unsqueeze(1))
            loss = points_render_loss(translations, sampled_neutral, target_img_batch)

            losses.append(loss.item())
            loss.backward()
            grad_norm = np.mean([x.grad.view(-1).abs().mean().cpu().numpy() for x in model.parameters()])
            norms.append(grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if iter_ind % 5 == 0 and iter_ind < 40:
            lr *= 1.1
            print(f'new lr = {lr}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # if iter_ind % 20 == 0 and 60 < iter_ind < 200:
        #     lr *= 0.5
        #     print(f'new lr = {lr}')
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        model.eval()

        elapsed = time.time() - start_time

        print(f'{iter_ind} loss = {np.mean(losses)}, grad_norm = {np.mean(norms)}, elapsed = {elapsed}')

        torch.save(model, f'saved_models/model_{iter_ind}.pt')

        err = 0.0
        n_errs = 0
        n_errs2 = 0
        for target_ind, (target_img, target_translate) in enumerate(zip(target_images, target_translations)):
            target_img = torch.FloatTensor(target_img)
            target_img = target_img.cuda()

            translation = model(target_img.unsqueeze(0).unsqueeze(0)).squeeze()
            cur_err = np.linalg.norm(translation.detach().cpu().numpy().round() - target_translate.round())
            err += cur_err
            if cur_err > 1:
                n_errs += 1
            if cur_err > 2:
                n_errs2 += 1
        print('err =', err / len(target_images), 'n_errs =', n_errs, 'n_errs2 =', n_errs2)

        for target_ind, (target_img, target_translate) in enumerate(
                itertools.islice(zip(target_images, target_translations), 18)):
            target_img = torch.FloatTensor(target_img)
            target_img = target_img.cuda()

            translation = model(target_img.unsqueeze(0).unsqueeze(0)).squeeze()
            points = sampled_neutral + translation

            # print(f'translation = {translation}')
            # print(f'translation target = {target_translate}')

            canvas = cv2.cvtColor(target_img.detach().cpu().numpy(), cv2.COLOR_GRAY2BGR)
            draw_points(points.detach().cpu().numpy(), canvas, (0, 255, 0), 0)
            cv2.imshow(f'canvas_{target_ind}', cv2.pyrUp(cv2.pyrUp(canvas)))

            cv2.waitKey(10)
    cv2.waitKey()


if __name__ == '__main__':
    # main1()
    main2()
