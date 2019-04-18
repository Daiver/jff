import torch


def function(x):
    return torch.stack((x[0] * 2, x[0] + x[1] * 5))


x = torch.FloatTensor([11, 7]).requires_grad_(True)
n_vars = 2
jac_list = []
for i in range(n_vars):
    y = function(x)
    print(y)
    grad_mult = torch.FloatTensor([1 if ii == i else 0 for ii in range(n_vars)])
    y.backward(grad_mult)
    jac_list.append(x.grad.clone())
    x.grad.zero_()
jac = torch.stack(jac_list)
print(jac)
