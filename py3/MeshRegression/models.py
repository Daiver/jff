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
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_final = nn.Dropout(0.5)
        self.fc1 = nn.Linear(n_final_features, n_moddle_features)
        self.fc_final = nn.Linear(n_moddle_features, n_out_vertices * 3)

    def forward(self, x):
        x = self.backbone.features(x)
        # print(x.shape)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.n_final_features)

        x = self.dropout_1(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout_final(x)
        x = self.fc_final(x)
        x = x.view(-1, self.n_out_vertices, 3)

        return x



