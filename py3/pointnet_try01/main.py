import numpy as np
import cv2
from copy import deepcopy
from scipy.spatial.transform import Rotation

import geom_tools
import np_draw_tools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import SpatialTransformer, PointNet
from dataset_tools import DatasetWithAugmentations


def draw_geom(geom: geom_tools.Mesh, canvas_size, transformation=None) -> np.ndarray:
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    vertices = geom.vertices
    if transformation is not None:
        vertices = geom_tools.transform_vertices(transformation, vertices)
    for v in vertices:
        cv2.circle(canvas, np_draw_tools.to_int_tuple(v[:2]), 1, (0, 255, 0), -1)
    return canvas


def main():
    path_to_geom = "/home/daiver/Downloads/R3DS_Wrap_3.4.8_Linux/Gallery/Blendshapes/Basemesh.obj"
    geom = geom_tools.load(path_to_geom)
    bbox = geom.bbox()
    print(geom_tools.summary(geom))
    print(bbox)

    # model = SpatialTransformer(3, 64)
    model = PointNet(3)
    # x = torch.zeros(2, 3, 10)
    # x = model(x)
    # print(x)
    # print(x.shape)

    canvas_size = (1024, 1024)
    view_transform = geom_tools.fit_to_view_transform(geom.bbox(), canvas_size)

    # canvas = draw_geom(geom, canvas_size, view_transform)
    # cv2.imshow("", canvas)
    # cv2.waitKey()

    n_epochs = 100
    batch_size = 4
    lr = 1e-3
    device = "cuda"
    # device = "cpu"

    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

    dataset = [
        deepcopy(geom.vertices),
        deepcopy(geom.vertices),
        deepcopy(geom.vertices),
        deepcopy(geom.vertices),
    ]
    data_transformer = DatasetWithAugmentations(dataset)
    dataloader = DataLoader(
        dataset=data_transformer, num_workers=4, batch_size=batch_size
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        for iteration, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            res = model(data)
            loss = loss_function(res, target)
            loss.backward()
            print(f"{epoch + 1} / {n_epochs} loss = {loss.item()}")

            optimizer.step()
            optimizer.zero_grad()
        model.eval()


if __name__ == "__main__":
    main()
