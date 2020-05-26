import torch


def screen_to_norm(coords: torch.Tensor, width: int, height: int, align_corners: bool = True):
    assert align_corners
    assert coords.dim() == 2
    assert coords.shape[1] == 2
    assert width > 0 and height > 0

    res = coords.clone()
    res[:, 0] *= 2.0 / float(width - 1)
    res[:, 1] *= 2.0 / float(height - 1)
    res.sub_(1)
    return res


def screen_to_norm_batch(coords: torch.Tensor, width: int, height: int, align_corners: bool = True):
    assert align_corners
    assert coords.dim() == 3
    assert coords.shape[2] == 2
    assert width > 0 and height > 0

    res = coords.clone()
    res[:, :, 0] *= 2.0 / float(width - 1)
    res[:, :, 1] *= 2.0 / float(height - 1)
    res.sub_(1)
    return res

