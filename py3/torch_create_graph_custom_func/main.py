import numpy as np
import torch

class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        print('forward call')
        ctx.save_for_backward(input)
        return input * input

    @staticmethod
    def backward(ctx, grad_output):
        print('backward call')
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input =  grad_output * 2 * input
        return grad_input


v = torch.from_numpy(np.array([4.0], dtype=np.float32))
x = torch.FloatTensor((3,))
x.requires_grad=True
f = x
f = MySquare.apply(f)
#f = f[0] * f[0]
print('f=', f)
grad_f, = torch.autograd.grad(f, x, create_graph=True)
print('df/dx=', grad_f)
z = grad_f @ v
z.backward(retain_graph=True)
print('res=', x.grad)
print(torch.autograd.grad(grad_f, x))
