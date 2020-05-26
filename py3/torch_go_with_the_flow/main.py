import time
import numpy as np
import cv2

import np_draw_tools

import utils
from utils import visualize_optical_flow, warp_by_flow

canvas_size = (128, 128)


def draw_sample(coord):
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
    x, y = np.array(coord).round().astype(np.int32)
    cv2.circle(canvas, center=(x, y), radius=10, color=(0, 255, 0), thickness=-1)
    return canvas


def main():
    img0 = draw_sample((50, 50))
    img1 = draw_sample((75, 50))

    time_start = time.time()
    flow = utils.compute_flow_for_clip([img0, img1])[0]
    print(f"elapsed {time.time() - time_start}")

    warp = warp_by_flow(img1, flow)

    flow_vis_rgb = visualize_optical_flow(flow)

    cv2.imshow("i0_w", warp)
    cv2.imshow("img0", img0)
    cv2.imshow("img1", img1)
    cv2.imshow("flow", flow_vis_rgb)
    np_draw_tools.wait_esc()


if __name__ == '__main__':
    main()


# import torch
