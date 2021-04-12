import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(
            self,
            n_features: int,
            n_heads: int
    ):
        super().__init__()
        self.n_features = n_features
        self.n_heads = n_heads

        self.tokeys = nn.Linear(n_features, n_features * n_heads, bias=False)
        self.toqueries = nn.Linear(n_features, n_features * n_heads, bias=False)
        self.tovalues = nn.Linear(n_features, n_features * n_heads, bias=False)
        self.unifyheads = nn.Linear(n_features * n_heads, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_size, n_features = x.shape
        assert n_features == self.n_features
        keys = self.tokeys(x).view(batch_size, seq_size, self.n_heads, n_features)
        queries = self.toqueries(x).view(batch_size, seq_size, self.n_heads, n_features)
        values = self.tovalues(x).view(batch_size, seq_size, self.n_heads, n_features)

        keys = keys.transpose(1, 2).view(batch_size * self.n_heads, seq_size, n_features)
        queries = queries.transpose(1, 2).view(batch_size * self.n_heads, seq_size, n_features)
        values = values.transpose(1, 2).view(batch_size * self.n_heads, seq_size, n_features)

        queries = queries / (n_features ** (1/4))
        keys = keys / (n_features ** (1/4))

        # queries : batch_size*n_heads x seq_size x n_features
        scores = torch.bmm(queries, keys.transpose(1, 2))
        # scores : batch_size*n_heads x seq_size x seq_size ->
        # scores : batch_size*n_heads x seq_size x seq_size
        scores = torch.softmax(scores, dim=2)

        # scores : batch_size*n_heads x seq_size x seq_size
        # values : batch_size*n_heads x seq_size x n_features
        res = torch.bmm(scores, values).view(batch_size, self.n_heads, seq_size, n_features)
        res = res.transpose(1, 2).view(batch_size, seq_size, self.n_heads * n_features)
        return self.unifyheads(res)
