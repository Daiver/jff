import numpy as np
import torch

x = torch.from_numpy(np.array([
    [1, 0],
    [0, 1]
], dtype=np.float))
x.requires_grad_(True)

y = torch.from_numpy(np.array([
    [0, 2],
    [1, 0]
], dtype=np.float))

cov_mat = x.transpose(0, 1) @ y

print(f"cov_mat cov_mat.requires_grad = {cov_mat.requires_grad}")
print(cov_mat)

# u, s, vt = torch.svd(cov_mat)
u, s, v = torch.svd(cov_mat)
vt = v.transpose(0, 1)

r = vt @ u.transpose(0, 1)
print(f"{vt.requires_grad} {u.requires_grad}")
print(f"r r.requires_grad = {r.requires_grad}")
print(r)

residual = (r - torch.tensor([
    [1, 0],
    [0, 1]
]).float()).reshape(-1, 1)
print("residual")
print(residual)
loss = torch.sum(residual.transpose(0, 1) @ residual)
print(f"loss = {loss.item()} loss.requires_grad={loss.requires_grad}")
loss.backward()

print("x.grad")
print(x.grad)
