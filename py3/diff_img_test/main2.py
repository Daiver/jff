import cv2

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from losses import mk_diff_img, loop_laplacian_loss


def main1():
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
        [1, 3],
        [3, 4],
    ])
    points = points.cuda()
    points.requires_grad_()

    for iter_num in range(10):
        loss = mod.apply(points).sum()
        print('loss =', loss.item())
        loss.backward()
        print('dx =', points.grad.view(-1))
        points.data.add_(-0.1 * points.grad)
        print('points =', points.detach().view(-1))
        points.grad.zero_()


def draw_points(points, img, color=255, scale=1.0):
    points = (scale * points.data).round().int()
    for p in points:
        cv2.circle(img, (p[0], p[1]), 0, color, -1)


# def gen_circle():
#     cv2.ellipse(img, center, axes, angle)


def gen_points(n_points, center, radius):
    res = np.zeros((n_points, 2), dtype=np.float32)
    angle_step = 2 * np.pi / float(n_points)
    for i in range(n_points):
        res[i] = (np.sin(angle_step * i), np.cos(angle_step * i))
    res *= radius
    res += center
    return res


def main2():
    img = np.zeros((32, 32), dtype=np.float32)
    cv2.circle(img, (16, 16), 8, 1.0, -1)
    # img_blurred = cv2.GaussianBlur(img, (15, 15), 1.5)
    img_blurred = cv2.GaussianBlur(img, (15, 15), 0.5)
    img_blurred = torch.from_numpy(img_blurred).cuda()

    # cv2.imshow('', cv2.pyrUp(cv2.pyrUp(img)))
    # cv2.waitKey()

    diff_img = mk_diff_img(img_blurred)

    # points_old = torch.FloatTensor([
    #     [13, 13],
    #     [19, 13],
    #     [19, 19],
    #     [13, 19],
    # ])
    # points_old = points_old.cuda()
    # points = torch.FloatTensor([
    #     [13, 13],
    #     [19, 13],
    #     [19, 19],
    #     [13, 19],
    # ])

    points = gen_points(64, (8, 8), 6)
    points = torch.FloatTensor(points)
    print(points)

    points = points.cuda()
    points_old = points.clone()
    points.requires_grad_()

    targets = (torch.ones(points.shape[0]) * img_blurred.max()).cuda()
    # targets = torch.ones(points.shape[0]).cuda()
    print('max val', img_blurred.max())

    img2 = cv2.cvtColor(cv2.pyrUp(cv2.pyrUp(img)), cv2.COLOR_GRAY2BGR)
    draw_points(points, img2, color=(0, 255, 0), scale=4)
    cv2.imshow('', cv2.pyrUp(cv2.pyrUp(img)))
    cv2.imshow('2', cv2.pyrUp(cv2.pyrUp(img_blurred.cpu().numpy())))
    cv2.imshow('3', img2)
    cv2.waitKey(1000)

    lrs_and_inner_iters = [
        (400, 1.0),
        (200, 0.1),
        (200, 0.01),
        (200, 0.001),
        (500, 0.0001),
        (1000, 0.00001)
    ]

    lr = lrs_and_inner_iters[0][1]
    optimizer = optim.Adam([points], lr=lr)

    n_outer_iters = len(lrs_and_inner_iters)
    for outer_iter in range(n_outer_iters):
        n_inner_iters = lrs_and_inner_iters[outer_iter][0]
        for iter_num in range(n_inner_iters):
            data_loss = (diff_img.apply(points) - targets).norm(2) / len(points)
            smooth_loss = loop_laplacian_loss(points, points_old)
            loss = 100 * data_loss + 1 * smooth_loss
            if iter_num % 10 == 0:
                print('lr =', lr, 'data =', data_loss.item(), 'smooth_loss =', smooth_loss.item(), 'loss =', loss.item())
            loss.backward()
            optimizer.step()
            # points.data.add_(-0.02 * points.grad)
            # print('dx =', points.grad.view(-1))
            # print('points =', points.detach().view(-1))
            points.grad.zero_()

            if iter_num % 100 == 0:
                img2 = cv2.cvtColor(cv2.pyrUp(cv2.pyrUp(img)), cv2.COLOR_GRAY2BGR)
                draw_points(points, img2, color=(0, 255, 0), scale=4)
                cv2.imshow('', cv2.pyrUp(cv2.pyrUp(img)))
                cv2.imshow('2', cv2.pyrUp(cv2.pyrUp(img_blurred.cpu().numpy())))
                cv2.imshow('1', img2)
                cv2.waitKey(10)

        lr = lrs_and_inner_iters[outer_iter][1]
        print(outer_iter, 'new lr', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print(points.data)

    # img2 = img.copy()
    # draw_points(points, img2)
    # cv2.imshow('', cv2.pyrUp(cv2.pyrUp(img)))
    # cv2.imshow('1', cv2.pyrUp(cv2.pyrUp(img2)))
    cv2.waitKey()


if __name__ == '__main__':
    # main1()
    main2()
