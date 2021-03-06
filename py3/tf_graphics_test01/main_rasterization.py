import time
import numpy as np
import cv2
import geom_tools
import tensorflow as tf
import tensorflow_graphics
from tensorflow_graphics.rendering import rasterizer


def draw_depth(depth_tf, min_depth):
    depth = depth_tf.numpy().reshape(depth_tf.shape[0], depth_tf.shape[1])
    depth[np.isinf(depth)] = min_depth
    min_val = depth.min()
    max_val = depth.max()
    delta = max_val - min_val
    if abs(delta) < 1e-10:
        delta = 1.0
    depth -= min_val
    depth /= (max_val - min_val)
    cv2.imshow("depth", depth)
    cv2.waitKey()


def main():
    tf.enable_eager_execution()
    teapot_model = geom_tools.load("teapot.obj")
    geom_tools.summary(teapot_model)
    print(teapot_model.bbox())
    print(teapot_model.bbox().size())

    faces_tf = tf.convert_to_tensor(teapot_model.triangle_vertex_indices)
    vertices_tf = tf.convert_to_tensor(teapot_model.vertices, dtype=tf.float32)

    vertices_tf = vertices_tf * 20 + [0, 0, 2.2]

    start_time = time.time()
    depth, tri_indices, bary = rasterizer.rasterize(vertices_tf, faces_tf, 256, 256)
    print("elapsed", time.time() - start_time)
    draw_depth(depth, -1e-5)


if __name__ == '__main__':
    main()
