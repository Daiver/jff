import numpy as np
import torch
import torch.nn.functional
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.utils.data import DataLoader

from models import RNN, hidden_size


def main():
    batch_size = 4
    n_epochs = 10000

    data = torch.tensor([
        [1, 2, 0, 3],
        [4, 4, 2, 1],
        [1, 1, 1, 1],
        [1, 3, 1, 1],
        [0, 1, 3, 0]
    ])
    targets = torch.flip(data, [1])

    data = nnf.one_hot(data).float()
    targets = nnf.one_hot(targets).float()
    pairs = list(zip(data, targets))

    seq_len = data.shape[1]
    n_features = data.shape[2]

    dataloader = DataLoader(dataset=pairs, batch_size=batch_size, shuffle=True)
    model = RNN(input_size=n_features, output_size=n_features)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # criterion = nn.NLLLoss()
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()

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
            if epoch == n_epochs - 1:
                print(torch.argmax(label, -1))
                print(torch.argmax(res, -1))

            loss = criterion(res, label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        # if epoch % 10 == 0:
        print(f"{epoch + 1:04d}/{n_epochs} {np.mean(losses)}")


if __name__ == '__main__':
    main()

