import random
import itertools
import time
import os
import cv2

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from losses import mk_diff_img
from models import Model

from train3 import gen_lines, sample_lines, default_line_scale, default_line_offset, sampling_rate, draw_points


def main():
    neutral_lines = gen_lines(default_line_offset, default_line_scale)
    sampled_neutral = sample_lines(neutral_lines, sampling_rate)
    sampled_neutral = torch.FloatTensor(sampled_neutral)

    target_images = [
        cv2.imread(os.path.join('tmp/', x), 0)
        for x in os.listdir('tmp')
        if x.endswith('.png')
    ]

    # model = Model()
    # model.load_state_dict(torch.load('saved_models/model_500.pt'))
    # model.eval()
    model = torch.load('saved_models/model_500.pt').cpu()
    print(model)

    print(len(target_images))
    for target_ind, target_img_orig in enumerate(target_images):

        target_img = torch.FloatTensor(target_img_orig)
        translation = model(target_img.unsqueeze(0).unsqueeze(0))

        points = sampled_neutral + translation
        # print(f'translation = {translation}')
        # print(f'translation target = {target_translate}')

        canvas = cv2.cvtColor(target_img_orig, cv2.COLOR_GRAY2BGR)
        draw_points(points.detach().cpu().numpy(), canvas, (0, 255, 0), 0)
        cv2.imwrite(f'res/{target_ind}.png', canvas)
        # cv2.imshow('', cv2.pyrUp(cv2.pyrUp(target_img_orig)))
        # cv2.imshow('res', cv2.pyrUp(cv2.pyrUp(canvas)))
        # cv2.waitKey()


if __name__ == '__main__':
    main()
