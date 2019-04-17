import torch
import torch.nn.functional as F


def channelwise_conv(torch_img, kernel):
    pass


def img_grad_dx(torch_img):
    assert len(torch_img.shape) == 3
    torch_img = torch_img.unsqueeze(0)
    assert False
    # kernel =
