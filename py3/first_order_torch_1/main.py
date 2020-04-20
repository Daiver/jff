from typing import Union, List, Tuple
import random
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import np_draw_tools

from models import KpPredictor

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def draw_background(canvas_size: Union[List[int], Tuple[int, int]]):
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    scale = canvas_size[0] / 64.0 * 0.5
    # cv2.putText(
    #     canvas, "A",
    #     (canvas_size[0] // 2, canvas_size[1] // 2),
    #     cv2.FONT_HERSHEY_COMPLEX, scale, (0.5, 0.5, 0.5))
    #
    cv2.putText(
        canvas, "B",
        (canvas_size[0] // 2 - canvas_size[0] // 3, canvas_size[1] // 2 + canvas_size[1] // 4),
        cv2.FONT_HERSHEY_COMPLEX, scale, (0.0, 0.7, 0.5))
    return canvas


def draw_fig1_on_background(background: np.ndarray, pos) -> np.ndarray:
    canvas = background.copy()
    points = np.array([
        [30, 30],
        [30, 50],
        [50, 30]
    ], dtype=np.float32)
    points += pos
    points = points.round().astype(np.int32).reshape((1, -1, 2))
    cv2.fillPoly(canvas, points, color=(0, 0, 1))
    return canvas


def mk_translation_sequence(background: np.ndarray, n_samples: int, samples_maker) -> np.ndarray:
    random_translations = np.random.uniform(-20, 20, size=(n_samples, 2))
    return np.array([
        samples_maker(background, pos)
        for pos in random_translations
    ], dtype=np.float32)


def torch_translate_images(images, translations, device):
    n_samples = images.shape[0]
    assert translations.shape[0] == n_samples
    rotations = torch.from_numpy(np.tile(np.eye(2, dtype=np.float32), (n_samples, 1, 1))).to(device)

    translations = translations.view(-1, 2, 1)
    thetas = torch.cat((rotations, translations), 2)
    grid = F.affine_grid(thetas, size=images.shape)

    res = F.grid_sample(images, grid)
    return res


def main():
    canvas_size = (64, 64)
    n_samples = 2048
    device = "cuda"
    batch_size = 32
    n_epochs = 100
    lr = 1e-3

    background = draw_background(canvas_size)
    fig1 = draw_fig1_on_background(background, [10, 10])
    train_figs = mk_translation_sequence(background, n_samples, draw_fig1_on_background)

    grid = np_draw_tools.make_grid(train_figs[:50], 4)

    cv2.imshow("back", background)
    cv2.imshow("fig1", grid)
    cv2.waitKey(100)

    train_dataset = torch.from_numpy(train_figs).permute([0, 3, 1, 2])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    kp_predictor = KpPredictor().to(device)
    optimizer = optim.Adam(kp_predictor.parameters(), lr=lr)

    for epoch_ind in range(n_epochs):
        losses = []
        for batch_ind, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            translations = kp_predictor(batch)

            n_images = batch.shape[0]
            assert n_images % 2 == 0
            divide_ind = n_images // 2
            base_images = batch[:divide_ind]
            target_images = batch[divide_ind:]

            base_translations = translations[:divide_ind]
            target_translations = translations[divide_ind:]

            final_translations = target_translations - base_translations

            transformed_images = torch_translate_images(base_images, final_translations, device)

            loss = F.l1_loss(transformed_images, target_images)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"{epoch_ind + 1}/{n_epochs} loss: {np.mean(losses)}")

    test_figs = mk_translation_sequence(background, n_samples, draw_fig1_on_background)
    test_dataset = torch.from_numpy(test_figs).permute([0, 3, 1, 2])

    images = test_dataset[:20].to(device)
    translations = kp_predictor(images)

    n_images = images.shape[0]
    assert n_images % 2 == 0
    divide_ind = n_images // 2
    base_images = images[:divide_ind]
    target_images = images[divide_ind:]
    base_translations = translations[:divide_ind]
    target_translations = translations[divide_ind:]
    final_translations = target_translations - base_translations
    transformed_images = torch_translate_images(base_images, final_translations, device)

    for x, y in zip(target_images, transformed_images):
        x = x.cpu().detach().permute([1, 2, 0]).numpy()
        y = y.cpu().detach().permute([1, 2, 0]).numpy()
        diff = np.abs(x - y)

        cv2.imshow("x", x)
        cv2.imshow("y", y)
        cv2.imshow("d", diff)
        cv2.waitKey()


if __name__ == '__main__':
    main()

