import math
import time
import torch

# Our module!
import lltm_cpp
print(lltm_cpp)


class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cpp.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)


print("Before call")
lltm_cpp.foo()
print(lltm_cpp.barycoords_from_2d_trianglef(
    0, 0,
    1, 0,
    0, 1,
    0.5, 0.5))

# batch_size = 16
# input_features = 32
# state_size = 128
#
# X = torch.randn(batch_size, input_features)
# h = torch.randn(batch_size, state_size)
# C = torch.randn(batch_size, state_size)
#
# rnn = LLTM(input_features, state_size)
#
# forward = 0
# backward = 0
# for _ in range(100000):
#     start = time.time()
#     new_h, new_C = rnn(X, (h, C))
#     forward += time.time() - start
#
#     start = time.time()
#     (new_h.sum() + new_C.sum()).backward()
#     backward += time.time() - start
#
# print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))
