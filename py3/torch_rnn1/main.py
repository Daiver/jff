import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.utils.data import DataLoader


hidden_size = 3


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 1
        output_size = 1
        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        #
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        inner_size = 8
        self.i2h = nn.Sequential(
            nn.Linear(input_size + hidden_size, inner_size),
            nn.LeakyReLU(),
            nn.Linear(inner_size, hidden_size),
        )

        self.i2o = nn.Sequential(
            nn.Linear(input_size + hidden_size, inner_size),
            nn.LeakyReLU(),
            nn.Linear(inner_size, output_size),
        )

    def forward(self, inp: torch.Tensor, hidden: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        combined = torch.cat((inp, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = torch.sigmoid(output)
        return output, hidden


seq_len = 8


def int2uint4(val: int) -> np.ndarray:
    res = np.zeros(seq_len, dtype=np.float32)
    res[7] = val % 2
    val //= 2
    res[6] = val % 2
    val //= 2
    res[5] = val % 2
    val //= 2
    res[4] = val % 2
    val //= 2
    res[3] = val % 2
    val //= 2
    res[2] = val % 2
    val //= 2
    res[1] = val % 2
    val //= 2
    res[0] = val % 2
    return res


def main():
    n_epochs = 50000
    batch_size = 32

    data = [
        int2uint4(x)
        for x in range(256)
    ]
    labels = [
        1.0 if i % 5 == 0 else 0.0
        # 1.0 if i % 4 == 0 else 0.0
        for i in range(len(data))
    ]
    pairs = list(zip(labels, data))
    for i, (label, x) in enumerate(pairs):
        print(i, label, x)

    dataloader = DataLoader(dataset=pairs, batch_size=batch_size, shuffle=True)
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # criterion = nn.NLLLoss()
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        losses = []
        for batch in dataloader:
            label, x = batch
            label = label.unsqueeze(-1).float()
            x = x.permute(1, 0).view(seq_len, -1, 1)
            hidden = torch.zeros(x.shape[1], hidden_size)
            for i in range(seq_len):
                res, hidden = model(x[i], hidden)
            loss = criterion(res, label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        if epoch % 10 == 0:
            print(f"{epoch + 1:04d}/{n_epochs} {np.mean(losses)}")

    model.eval()
    for batch in dataloader:
        label, x = batch
        label = label.unsqueeze(-1).float()
        x = x.permute(1, 0).view(seq_len, -1, 1)
        hidden = torch.zeros(x.shape[1], hidden_size)
        for i in range(seq_len):
            res, hidden = model(x[i], hidden)
        res = res.detach().round()
        loss = nnf.mse_loss(res, label)
        res = res.round()
        print(label.squeeze())
        print(res.squeeze())
        print(loss.squeeze())


if __name__ == '__main__':
    main()
