import numpy as np
np.set_printoptions(edgeitems=50, linewidth=175)
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    img = np.zeros((7, 7), dtype=np.uint8)
    img[:] = 1
    img[2, 2] = 0
    img[3, 3] = 0
    #img[4, 3] = 0
    print(img)
    #distField = cv2.distanceTransform(img, cv2.DIST_L1, 0)
    distField = cv2.distanceTransform(img, cv2.DIST_L2, 0)
    print(distField)

    dxKernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32) / 9.0
    dx = cv2.filter2D(distField, -1, dxKernel)
    print('Dx')

    dyKernel = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=np.float32) / 9.0
    dy = cv2.filter2D(distField, -1, dyKernel)
    print('Dy')

    print(dx)

    plt.imshow(dx)
    plt.show()

    print(dy)
    plt.imshow(dy)
    plt.show()
