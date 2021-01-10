import numpy as np
import torch
import torch.nn.functional
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.utils.data import DataLoader

from models import RNN, hidden_size


def main():
    batch_size = 2
    n_epochs = 1000

    data = torch.tensor([
        [1, 2, 0, 3],
        [4, 5, 2, 1],
    ])
    targets = torch.flip(data, [1])
    data = nnf.one_hot(data).float()
    targets = nnf.one_hot(targets).float()
    pairs = list(zip(data, targets))

    seq_len = data.shape[1]
    n_features = data.shape[2]

    dataloader = DataLoader(dataset=pairs, batch_size=batch_size, shuffle=True)
    model = RNN(input_size=n_features, output_size=n_features)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.NLLLoss()
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        losses = []
        for batch in dataloader:
            label, x = batch

            hidden = torch.zeros(x.shape[0], hidden_size)
            ress = []
            for i in range(seq_len):
                res, hidden = model(x[:, i], hidden)
                ress.append(res)
            res = torch.stack(ress, 1)

            loss = criterion(res, label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        # if epoch % 10 == 0:
        print(f"{epoch + 1:04d}/{n_epochs} {np.mean(losses)}")

    ress = []
    x = data
    hidden = torch.zeros(x.shape[0], hidden_size)
    for i in range(seq_len):
        res, hidden = model(x[:, i], hidden)
        ress.append(res)
    res = torch.stack(ress, 1)
    for x, y in zip(data, res):
        x = torch.argmax(x, dim=-1)
        y = torch.argmax(y, dim=-1)
        print(x.detach().numpy())
        print(y.detach().numpy())
        print("======")


if __name__ == '__main__':
    main()

