import multiprocessing
import cv2
import numpy as np


def visualize_optical_flow(flow: np.ndarray, out_pixel_type=np.float32) -> np.ndarray:
    assert flow.ndim == 3
    assert flow.shape[-1] == 2

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_vis_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    flow_vis_hsv[..., 1] = 255
    flow_vis_hsv[..., 0] = ang * 180 / np.pi / 2
    flow_vis_hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis_rgb = cv2.cvtColor(flow_vis_hsv, cv2.COLOR_HSV2BGR)
    # print(flow_vis_rgb.max(), flow_vis_rgb.min())
    return flow_vis_rgb.astype(out_pixel_type)


def warp_by_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    remap_flow = np.zeros((flow.shape[0], flow.shape[1], 2), dtype=np.float32)
    for row in range(flow.shape[0]):
        remap_flow[row, :, 1] = row
    for col in range(flow.shape[1]):
        remap_flow[:, col, 0] = col
    remap_flow += flow

    warp = cv2.remap(
        img,
        remap_flow, map2=None,
        interpolation=cv2.INTER_CUBIC)

    return warp


def compute_flow(img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    flow = cv2.optflow.calcOpticalFlowSF(
        img0.astype(np.uint8), img1.astype(np.uint8),
        layers=7, averaging_block_size=5, max_flow=5
    )

    flow[np.isnan(flow)] = 0.0
    if np.any(np.isnan(flow)):
        print("flow contains nan")
    return flow


def compute_flow_tuple(imgs: (np.ndarray, np.ndarray)) -> np.ndarray:
    return compute_flow(imgs[0], imgs[1])


def compute_flow_for_clip(images: [np.ndarray]) -> [np.ndarray]:
    n_images = len(images)
    assert n_images > 1

    pool = multiprocessing.Pool()
    pairs = []
    for i in range(0, n_images - 1):
        pairs.append((images[i], images[i + 1]))
    res = pool.map(compute_flow_tuple, pairs)

    return res
