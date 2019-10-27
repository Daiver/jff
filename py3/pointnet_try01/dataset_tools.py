from typing import Collection
import numpy as np
from scipy.spatial.transform.rotation import Rotation
import geom_tools


class DatasetWithAugmentations:
    def __init__(self, dataset: Collection):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        angle = np.random.uniform(-np.pi / 4.0, np.pi / 4.0)
        rot = Rotation.from_euler("z", angle, degrees=False)
        transformation = geom_tools.rotation_around_vertex(rotation_matrix=rot.as_dcm(), rotation_center=sample.mean())
        sample = geom_tools.transform_vertices(transformation)
        return sample, angle
