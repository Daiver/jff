import torch
from torch import nn as nn


hidden_size = 4


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 4
        output_size = 4

        inner_size = 16
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
