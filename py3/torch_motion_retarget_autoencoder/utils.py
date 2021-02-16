import torch


def numpy_images_to_torch(images):
    return [
        torch.from_numpy(x).float().permute(2, 0, 1) / 255.0
        for x in images
    ]