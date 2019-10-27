from typing import Collection
import numpy as np
from scipy.spatial.transform.rotation import Rotation
import geom_tools


def deform_sample(sample: np.ndarray, angle: float):
    rot = Rotation.from_euler("z", angle, degrees=False).as_dcm()
    transformation = geom_tools.rotation_around_vertex(rotation_matrix=rot, rotation_center=sample.mean(axis=0))
    sample = geom_tools.transform_vertices(transformation, sample)
    return sample


class DatasetWithAugmentations:
    def __init__(self, dataset: Collection):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        angle = np.random.uniform(-np.pi / 4.0, np.pi / 4.0)
        sample = deform_sample(sample, angle)
        sample = sample.transpose()
        return sample, np.array([angle], dtype=np.float32)
