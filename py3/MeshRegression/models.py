import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, backbone, n_final_features, n_out_vertices):
        super().__init__()
        self.backbone = backbone
        self.n_final_features = n_final_features
        self.n_out_vertices = n_out_vertices
        self.add_module("backbone", backbone)
        self.dropout = nn.Dropout(0.1)
        self.fc_final = nn.Linear(n_final_features, n_out_vertices * 3)

    def forward(self, x):
        x = self.backbone.features(x)
        #print(x.shape)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.n_final_features)

        x = self.dropout(x)
        x = self.fc_final(x)
        x = x.view(-1, self.n_out_vertices, 3)

        return x


class Model2(nn.Module):
    def __init__(self, backbone, n_final_features, n_moddle_features, n_out_vertices):
        super().__init__()
        self.backbone = backbone
        self.n_final_features = n_final_features
        self.n_out_vertices = n_out_vertices
        self.add_module("backbone", backbone)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_final = nn.Dropout(0.5)
        self.fc1 = nn.Linear(n_final_features, n_moddle_features)
        self.fc_final = nn.Linear(n_moddle_features, n_out_vertices * 3)

    def forward(self, x):
        x = self.backbone.features(x)
        # print(x.shape)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.n_final_features)

        x = self.dropout_1(x)
        x = F.relu(self.fc1(x))
        x = self.fc_final(x)
        x = x.view(-1, self.n_out_vertices, 3)

        return x


class VanillaFinBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class BNFinBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(),
            nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class ResidualFinBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.PReLU(),
            nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.PReLU(),
        )
        if stride != 0:
            self.shortcut = nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1)
        else:
            self.shortcut = None

    def forward(self, x):
        z = self.model(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + z


class FinNet(nn.Module):
    def __init__(self, n_middle_features, n_out_vertices, dropout_val=0.2):
        super().__init__()
        self.n_out_vertices = n_out_vertices

        # block = VanillaFinBlock
        # block = BNFinBlock
        block = ResidualFinBlock
        self.feature_extractor = nn.Sequential(
            block(1, 64, stride=2),
            block(64, 96, stride=2),
            block(96, 144, stride=2),
            block(144, 216, stride=2),
            block(216, 324, stride=2),
            block(324, 486, stride=2),
            # block(486, 512, stride=2),
        )

        self.n_flat_features = 486 * 5 * 4
        # self.n_flat_features = 512 * 3 * 3
        # self.n_flat_features = 486
        self.dropout = nn.Dropout(dropout_val)
        self.fc1 = nn.Linear(self.n_flat_features, n_middle_features)
        self.fc_final = nn.Linear(n_middle_features, 3 * n_out_vertices)

    def forward(self, x):
        x = self.feature_extractor(x)

        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, self.n_flat_features)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc_final(x)
        x = x.view(-1, self.n_out_vertices, 3)

        return x

