import os
import cv2

from utils import read_obj_vertices
import paths


class ImageToMeshDataset:
    def __init__(self, images, meshes, transform=None):
        assert len(images) == len(meshes)
        self.images = images
        self.meshes = meshes
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.transform is not None:
            return self.transform(self.images[item], self.meshes[item])
        return self.images[item], self.meshes[item]


def mk_kostet_dataset(indices: list=None, transform=None):
    images_root = os.path.join(paths.data_root, "KostetCentralResized")
    meshes_root = os.path.join(paths.data_root, "KostetSmoothCutted")

    image_name_pattern = "Mesh{:03d}.obj.png"
    mesh_name_pattern = "Object{:03d}.obj"

    if indices is None:
        indices = list(range(0, 299))

    images = [
        cv2.imread(os.path.join(images_root, image_name_pattern.format(idx)))
        for idx in indices
    ]

    meshes = [
        read_obj_vertices(os.path.join(meshes_root, mesh_name_pattern.format(idx)))
        for idx in indices
    ]

    n_vertices_first = len(meshes[0])
    for m in meshes:
        assert len(m) == n_vertices_first

    return ImageToMeshDataset(images, meshes, transform)


def mk_synth_dataset_train(indices: list=None, transform=None):
    images_root = os.path.join("/home/daiver/train_renders/")
    meshes_root = os.path.join("/home/daiver/train_geoms")

    image_name_pattern = "Mesh{:03d}.obj.png"
    mesh_name_pattern = "Object{:03d}.obj"

    if indices is None:
        indices = list(range(0, 1000))

    images = [
        cv2.imread(os.path.join(images_root, image_name_pattern.format(idx)))
        for idx in indices
    ]

    meshes = [
        read_obj_vertices(os.path.join(meshes_root, mesh_name_pattern.format(idx)))
        for idx in indices
    ]

    n_vertices_first = len(meshes[0])
    for m in meshes:
        assert len(m) == n_vertices_first

    return ImageToMeshDataset(images, meshes, transform)


def mk_synth_dataset_test(indices: list=None, transform=None):
    images_root = os.path.join("/home/daiver/test_renders/")
    meshes_root = os.path.join("/home/daiver/test_geoms")

    image_name_pattern = "Mesh{:03d}.obj.png"
    mesh_name_pattern = "Object{:03d}.obj"

    if indices is None:
        indices = list(range(0, 200))

    images = [
        cv2.imread(os.path.join(images_root, image_name_pattern.format(idx)))
        for idx in indices
    ]

    meshes = [
        read_obj_vertices(os.path.join(meshes_root, mesh_name_pattern.format(idx)))
        for idx in indices
    ]

    n_vertices_first = len(meshes[0])
    for m in meshes:
        assert len(m) == n_vertices_first

    return ImageToMeshDataset(images, meshes, transform)
