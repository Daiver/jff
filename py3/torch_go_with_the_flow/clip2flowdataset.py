import numpy as np
import torch


class Clip2FlowDataset:
    def __init__(self, images: [np.ndarray], flows: [np.ndarray]):
        self.n_items = len(flows)
        self.images = images
        self.flows = flows

        assert len(images) == self.n_items + 1

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        t_img1 = torch.from_numpy(self.images[idx]).permute([2, 0, 1]).float()
        t_img2 = torch.from_numpy(self.images[idx + 1]).permute([2, 0, 1]).float()
        t_flow = torch.from_numpy(self.flows[idx]).permute([2, 0, 1]).float()

        return t_img1, t_img2, t_flow
