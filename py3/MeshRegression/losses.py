import torch
import torch.nn.functional as F


class L1L2Loss:
    def __init__(self):
        pass

    def __call__(self, x, y):
        assert x.shape == y.shape
        assert len(x.shape) == 3
        diff = x - y
        n_samples = x.size(0)
        n_vertices = x.size(1)
        res = torch.norm(diff, dim=2).sum() / (n_samples * n_vertices)
        return res
