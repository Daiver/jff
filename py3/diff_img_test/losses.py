import torch
import torch.autograd as autograd
import torch.nn.functional as F


def mk_diff_img(image, channels_first=True):
    assert channels_first
    assert len(image.shape) == 2

    image = image.unsqueeze(0).unsqueeze(0)

    x_kernel = torch.Tensor([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]])
    x_kernel = x_kernel.view((1, 1, 3, 3)).cuda()

    y_kernel = torch.Tensor([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]])
    y_kernel = y_kernel.view((1, 1, 3, 3)).cuda()

    # padded_image = F.pad(image, [1, 1, 1, 1], value=image.abs().max())
    padded_image = F.pad(image, [1, 1, 1, 1], value=0)

    diff_img_x = F.conv2d(padded_image, y_kernel)
    diff_img_y = F.conv2d(padded_image, x_kernel)

    image = image.squeeze(0).squeeze(0)
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


def loop_laplacian_loss(points_positions, old_positions):
    assert len(points_positions.shape) == 2
    assert len(old_positions.shape) == 2
    n_points = points_positions.shape[0]
    res = 0.0
    for i in range(n_points):
        next_point_index = (i + 1) % n_points

        old_1 = old_positions[i]
        old_2 = old_positions[next_point_index]
        old_edge = old_2 - old_1

        point_1 = points_positions[i]
        point_2 = points_positions[next_point_index]
        edge = point_2 - point_1

        res += (edge - old_edge).norm(2)
    return res
