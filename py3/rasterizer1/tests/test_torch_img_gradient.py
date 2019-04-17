import unittest

import numpy as np
import torch
import torch_img_gradient


class TorchImgGradient(unittest.TestCase):
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
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ])
        res = torch_img_gradient.img_grad_dx(img)
        ans = torch.FloatTensor([
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ])
        self.assertTrue((res - ans).norm() < 1e-6)


if __name__ == '__main__':
    unittest.main()
