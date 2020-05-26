import cv2
import numpy as np

from main import canvas_size


def visualize_optical_flow(flow: np.ndarray, out_pixel_type=np.float32) -> np.ndarray:
    assert flow.ndim == 3
    assert flow.shape[-1] == 2

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_vis_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    flow_vis_hsv[..., 1] = 255
    flow_vis_hsv[..., 0] = ang * 180 / np.pi / 2
    flow_vis_hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis_rgb = cv2.cvtColor(flow_vis_hsv, cv2.COLOR_HSV2BGR)
    print(flow_vis_rgb.max(), flow_vis_rgb.min())
    return flow_vis_rgb.astype(out_pixel_type)


def warp_by_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    remap_flow = np.zeros((canvas_size[1], canvas_size[0], 2), dtype=np.float32)
    for row in range(canvas_size[1]):
        remap_flow[row, :, 1] = row
    for col in range(canvas_size[0]):
        remap_flow[:, col, 0] = col
    remap_flow += flow

    warp = cv2.remap(
        img,
        remap_flow, map2=None,
        interpolation=cv2.INTER_CUBIC)

    return warp
