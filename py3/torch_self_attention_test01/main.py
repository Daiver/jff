import torch
import torch.nn.functional
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.utils.data import DataLoader

from models import RNN


def mk_targets(dataset: torch.Tensor) -> torch.Tensor:
    assert dataset.ndim >= 2
    res = torch.clone(dataset)
    n_items_in_sample = dataset.shape[1]
    half_of_items = n_items_in_sample // 2
    res[half_of_items:] = dataset[:half_of_items]
    return res


def main():
    batch_size = 2
    n_epochs = 10

    data = torch.tensor([
        [1, 2, 0, 3],
        [4, 5, 2, 1],
    ])
    targets = torch.flip(data, [1])
    pairs = list(zip(nnf.one_hot(data), nnf.one_hot(targets)))

    dataloader = DataLoader(dataset=pairs, batch_size=batch_size, shuffle=True)
    model = RNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # criterion = nn.NLLLoss()
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        losses = []
        for batch in dataloader:
            label, x = batch


if __name__ == '__main__':
    main()

