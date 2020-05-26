import unittest

import numpy as np

import torch
import torch.nn.functional as F

from torch_tools import screen_to_norm, screen_to_norm_batch


class TestTorchTools(unittest.TestCase):
    def test_screen_to_norm01(self):
        width = 10
        height = 5
        coords = torch.from_numpy(np.array([
            [0, 0],
            [9, 4],
            [4.5, 2],
            [9, 0]
        ], dtype=np.float32))

        res = screen_to_norm(coords, width, height, align_corners=True)
        ans = torch.from_numpy(np.array([
            [-1, -1],
            [1, 1],
            [0, 0],
            [1, -1]
        ], dtype=np.float32))
        self.assertTrue(torch.allclose(ans, res))

    def test_screen_to_norm_batch01(self):
        width = 10
        height = 5
        coords = torch.from_numpy(np.array([
            [
                [0, 0],
                [9, 4],
            ],
            [
                [4.5, 2],
                [9, 0]
            ],
        ], dtype=np.float32))

        res = screen_to_norm_batch(coords, width, height, align_corners=True)
        ans = torch.from_numpy(np.array([
            [
                [-1, -1],
                [1, 1],
            ],
            [
                [0, 0],
                [1, -1]
            ],
        ], dtype=np.float32))
        self.assertTrue(torch.allclose(ans, res))
