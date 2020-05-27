import numpy as np
import torch


class Image2PointsDataset:
    def __init__(self, images: [np.ndarray], points: [np.ndarray]):
        self.n_items = len(images)
        self.images = images
        self.points = points

        assert len(points) == self.n_items

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        t_img = torch.from_numpy(self.images[idx]).permute([2, 0, 1]).float()
        t_position = torch.from_numpy(np.array(self.points[idx], dtype=np.float32)).float()

        return t_img, t_position
