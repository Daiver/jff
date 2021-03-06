import unittest

import numpy as np
import torch
import torch_img_gradient


class TestTorchImgGradient(unittest.TestCase):
    def test_img_dx01(self):
        img = torch.FloatTensor([
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ])
        res = torch_img_gradient.img_grad_dx(img)
        ans = torch.FloatTensor([
            [
                [0, 0, 0],
                [1, 0, -1],
                [0, 0, 0],
            ],
        ])
        self.assertTrue((res - ans).norm() < 1e-6)

    def test_img_dx02(self):
        img = torch.FloatTensor([
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ])
        res = torch_img_gradient.img_grad_dx(img)
        ans = torch.FloatTensor([
            [
                [0, 0, 0],
                [1, 0, -1],
                [0, 0, 0],
            ],
            [
                [0, -1, 0],
                [1, 0, -1],
                [0, 0, 0],
            ],
        ])
        self.assertTrue((res - ans).norm() < 1e-6)

    def test_img_dx03(self):
        img = torch.FloatTensor([
            [
                [0, 6, 0, 0],
                [0, -2, 3, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [5, 1, 2, 3],
            ],
        ])
        res = torch_img_gradient.img_grad_dx(img)
        ans = torch.FloatTensor([
            [
                [6, 0, -6, 0],
                [-2, 3, 2, -3],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, -3, 2, -2],
            ],
        ])
        self.assertTrue((res - ans).norm() < 1e-6)

    def test_img_dy01(self):
        img = torch.FloatTensor([
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ])
        res = torch_img_gradient.img_grad_dy(img)
        ans = torch.FloatTensor([
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, -1, 0],
            ],
        ])
        self.assertTrue((res - ans).norm() < 1e-6)


if __name__ == '__main__':
    unittest.main()
