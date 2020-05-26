import numpy as np
import cv2

import np_draw_tools


canvas_size = (128, 128)


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


def draw_sample(coord):
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
    x, y = np.array(coord).round().astype(np.int32)
    cv2.circle(canvas, center=(x, y), radius=10, color=(0, 255, 0), thickness=-1)
    return canvas


def main():
    i0 = draw_sample((50, 50))
    i1 = draw_sample((75, 50))
    # i1 = draw_sample((50, 50))

    # flow_searcher = cv2.optflow.createOptFlow_DeepFlow()
    # flow_searcher = cv2.createOptFlow_DualTVL1()
    # flow = np.zeros((canvas_size[1], canvas_size[0], 2), dtype=np.uint8)
    # flow = flow_searcher.calc(
    #     cv2.cvtColor(i0.astype(np.uint8), cv2.COLOR_BGR2GRAY),
    #     cv2.cvtColor(i1.astype(np.uint8), cv2.COLOR_BGR2GRAY),
    #     None
    # )
    # flow = cv2.calcOpticalFlowFarneback(
    #     cv2.cvtColor(i0.astype(np.uint8), cv2.COLOR_BGR2GRAY),
    #     cv2.cvtColor(i1.astype(np.uint8), cv2.COLOR_BGR2GRAY),
    #     None, 0.5, 3, 15, 3, 5, 1.2, 0
    # )
    flow = cv2.optflow.calcOpticalFlowSF(
        i0.astype(np.uint8), i1.astype(np.uint8),
        layers=7, averaging_block_size=7, max_flow=10
    )

    warp = warp_by_flow(i1, flow)

    flow_vis_rgb = visualize_optical_flow(flow)

    cv2.imshow("i0_w", warp)
    cv2.imshow("i0", i0)
    cv2.imshow("i1", i1)
    cv2.imshow("flow", flow_vis_rgb)
    np_draw_tools.wait_esc()


if __name__ == '__main__':
    main()


# import torch
