import torch
import torch.nn as nn
import torch.nn.functional as F


def mk_channelwise_conv_operator(kernel):
    conv_op = nn.Conv2d(
        in_channels=1, out_channels=1,
        kernel_size=kernel.shape,
        stride=1,
        padding=(kernel.shape[0] // 2, kernel.shape[1] // 2),
        bias=False)
    kernel = kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1])
    conv_op.weight.data = kernel
    return conv_op


def channelwise_conv(torch_img, kernel):
    assert len(torch_img.shape) == 4
    n_samples, n_channels, n_rows, n_cols = torch_img.shape
    torch_img_reshaped = torch_img.view(n_samples * n_channels, 1, n_rows, n_cols)
    conv_op = mk_channelwise_conv_operator(kernel=kernel)
    res = conv_op(torch_img_reshaped)
    res = res.view(n_samples, n_channels, n_rows, n_cols)
    return res


def img_grad_dx(torch_img):
    assert len(torch_img.shape) == 3
    torch_img = torch_img.unsqueeze(0)
    assert False
    # kernel =
