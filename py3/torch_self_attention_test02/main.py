import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from self_attention import SelfAttention
from transformer_block import TransformerBlock


def add_naive_position_encoding(input: torch.Tensor) -> torch.Tensor:
    res = torch.zeros(input.shape[0], input.shape[1], input.shape[2] + 1)
    res[:, :, 0 : input.shape[2]] = input
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            res[i, j, -1] = j / input.shape[1]
    return res


class Model(nn.Module):
    def __init__(self, n_features: int, n_heads: int, n_blocks: int):
        super(Model, self).__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(TransformerBlock(n_features=n_features, n_heads=n_heads))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
        # return torch.softmax(self.blocks(x), dim=2)


def main():
    train_input = [
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
    ]
    train_input = torch.tensor(train_input).float()
    train_input = add_naive_position_encoding(train_input)
    train_output = [
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        [
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
    ]
    train_output = torch.tensor(train_output).float()
    train_output = add_naive_position_encoding(train_output)

    batch_size = 1
    seq_size = 3
    n_features = 4
    model = Model(n_features=n_features, n_heads=4, n_blocks=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for iter_ind in range(2000):
        pred = model(train_input)
        loss = ((pred - train_output)**2).sum()
        print(f"{iter_ind} {loss.detach().item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pred = model(train_input)
    print(pred.round())


if __name__ == '__main__':
    main()

