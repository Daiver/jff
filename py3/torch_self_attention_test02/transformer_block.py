import torch
from torch import nn

from self_attention import SelfAttention


class Transformer(nn.Module):
    def __init__(
            self,
            n_features: int,
            n_heads: int
    ):
        super().__init__()
        self.attention = SelfAttention(n_features=n_features, n_heads=n_heads)

        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        x = self.norm1(x + attended)
        x2 = self.mlp(x)
        x = self.norm2(x + x2)
        return x
