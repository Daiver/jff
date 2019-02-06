import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def main():
    # Parameters and DataLoaders
    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                             batch_size=batch_size, shuffle=True)

    model = Model(input_size, output_size)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=[0, 1])

    model.to(device)

    for data in rand_loader:
        input = data.to(device)
        print(f"device = {device} inp.device = {input.device} "
              f"model.fc.weight.data.device = {model.module.fc.weight.data.device}")
        output = model(input)
        print("Outside: input size", input.size(),
              "output_size", output.size())


if __name__ == '__main__':
    main()
