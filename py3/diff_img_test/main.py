import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mk_diff_img(image, channels_first=True):
    assert channels_first
    assert len(image.shape) == 2

    image = image.unsqueeze(0).unsqueeze(0)

    x_kernel = torch.Tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
    x_kernel = x_kernel.view((1, 1, 3, 3)).cuda()

    y_kernel = torch.Tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]])
    y_kernel = y_kernel.view((1, 1, 3, 3)).cuda()

    padded_image = F.pad(image, [1, 1, 1, 1], value=image.abs().max())

    diff_img_x = F.conv2d(padded_image, y_kernel)
    diff_img_y = F.conv2d(padded_image, x_kernel)

    image.squeeze_(0)
    image.squeeze_(0)
    diff_img_x.squeeze_(0)
    diff_img_x.squeeze_(0)
    diff_img_y.squeeze_(0)
    diff_img_y.squeeze_(0)

    class LocalFunction(autograd.Function):
        def __init__(self):
            super().__init__()

        @staticmethod
        def forward(ctx, points_positions):
            assert len(points_positions.shape) == 2
            points_positions_detached = points_positions.detach().round().long()
            points_positions_detached[:, 0].clamp_(0, image.shape[0] - 1)
            points_positions_detached[:, 1].clamp_(0, image.shape[1] - 1)
            ctx.save_for_backward(points_positions_detached)
            return image[points_positions_detached[:, 0], points_positions_detached[:, 1]]

        @staticmethod
        def backward(ctx, grad_outputs):
            points_positions_detached, = ctx.saved_tensors

            d_x = diff_img_x[points_positions_detached[:, 0], points_positions_detached[:, 1]]
            d_y = diff_img_y[points_positions_detached[:, 0], points_positions_detached[:, 1]]
            res = torch.zeros(points_positions_detached.shape).cuda()

            res[:, 0] = grad_outputs * d_x
            res[:, 1] = grad_outputs * d_y
            return res

    return LocalFunction()


if __name__ == '__main__':
    img = torch.FloatTensor([
        [2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2],
        [2, 1, 0, 1, 2],
        [2, 1, 1, 1, 2],
        [2, 2, 2, 2, 2],
    ])
    img = img.cuda()
    mod = mk_diff_img(img)
    points = torch.FloatTensor([
        [0, 2],
        [1, 2],
        [2, 2],
    ])
    points = points.cuda()
    points.requires_grad_()

    for iter_num in range(10):
        loss = mod.apply(points).sum()
        print('loss =', loss.item())
        loss.backward()
        print('dx =', points.grad)
        points.data.add_(-0.1 * points.grad)
        print('points =', points)
        points.grad.zero_()
