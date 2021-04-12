import torch
from self_attention import SelfAttention


def main():
    batch_size = 2
    seq_size = 3
    n_features = 5
    att = SelfAttention(n_features=n_features, n_heads=4)
    x = torch.zeros(batch_size, seq_size, n_features)
    res = att(x)
    print(res.shape)


if __name__ == '__main__':
    main()

