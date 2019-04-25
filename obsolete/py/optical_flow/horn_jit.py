import sys
import cv2
import numpy as np
import numpy.linalg as lin
from numba import autojit, jit

@autojit
def takePixel(img, i, j):
    i = i if i >= 0 else 0
    j = j if j >= 0 else 0

    i = i if i < img.shape[0] else img.shape[0] - 1
    j = j if j < img.shape[1] else img.shape[1] - 1

    return img[i, j]

@autojit
def xDer(img, img2):
    res = np.zeros_like(img)
    for i in xrange(res.shape[0]):
        for j in xrange(res.shape[1]):
            sm = 0
            sm += takePixel(img, i, j + 1)      - takePixel(img, i, j)
            sm += takePixel(img, i + 1, j + 1)  - takePixel(img, i + 1, j)
            sm += takePixel(img2, i, j + 1)     - takePixel(img2, i, j)
            sm += takePixel(img2, i + 1, j + 1) - takePixel(img2, i + 1, j)
            sm /= 4.0
            res[i, j] = sm
    return res

@autojit
def yDer(img, img2):
    res = np.zeros_like(img)
    for i in xrange(res.shape[0]):
        for j in xrange(res.shape[1]):
            sm = 0
            sm += takePixel(img, i +1, j)       - takePixel(img, i, j)
            sm += takePixel(img, i + 1, j + 1)  - takePixel(img, i, j + 1)
            sm += takePixel(img2, i +1, j)      - takePixel(img2, i, j)
            sm += takePixel(img2, i + 1, j + 1) - takePixel(img2, i, j + 1)
            sm /= 4.0
            res[i, j] = sm
    return res

@autojit
def tDer(img, img2):
    res = np.zeros_like(img)
    for i in xrange(res.shape[0]):
        for j in xrange(res.shape[1]):
            sm = 0
            for ii in xrange(i, i + 2):
                for jj in xrange(j, j + 2):
                    sm += takePixel(img2, ii, jj) - takePixel(img, ii, jj)
            sm /= 4.0
            res[i, j] = sm
    return res


kernel = np.array([[ 0.08333333,  0.16666667,  0.08333333],
                   [ 0.16666667,  0.        ,  0.16666667],
                   [ 0.08333333,  0.16666667,  0.08333333]], dtype=np.float32)


@autojit
def average(img):
    return cv2.filter2D(img.astype(np.float32), 5, kernel)

'''@autojit
def average(img):
    res = np.zeros_like(img)
    for i in xrange(res.shape[0]):
        for j in xrange(res.shape[1]):
            sm = 0.0
            sm += 1.0/6.0 * (
                    takePixel(img, i - 1, j) +
                    takePixel(img, i + 1, j) +
                    takePixel(img, i, j - 1) +
                    takePixel(img, i, j + 1) 
                )
            sm += 1.0/12.0 * (
                    takePixel(img, i - 1, j - 1) +
                    takePixel(img, i + 1, j - 1) +
                    takePixel(img, i - 1, j + 1) +
                    takePixel(img, i + 1, j + 1) 
                )
            res[i, j] = sm

    return res
'''

@autojit
def solve(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    #Idt = img2 - img1
    #Idx = cv2.Sobel(img1, 5, 1, 0)
    #Idy = cv2.Sobel(img1, 5, 0, 1)
    Idx = xDer(img1, img2)
    Idy = yDer(img1, img2)
    Idt = tDer(img1, img2)

    alpha = 0.1
    u = np.zeros_like(img1)
    v = np.zeros_like(img1)
    ua = np.zeros_like(img1)
    va = np.zeros_like(img1)

    print( 'In iteration')

    b = 1.0 / (alpha**2 + Idx**2 + Idy**2)

    for i in xrange(20000):
        uo = u.copy()
        vo = v.copy()
        u = ua - Idx * (Idx * ua + Idy * va + Idt) * b#/(alpha**2 + Idx**2 + Idy**2)
        v = va - Idy * (Idx * ua + Idy * va + Idt) * b#/(alpha**2 + Idx**2 + Idy**2)

        ua = average(u)
        va = average(v)

        if np.allclose(u, uo) and np.allclose(v, vo):
            break
        #if i % 100 == 0:
        #    print (i)
    return u, v


if __name__ == '__main__':
    print ('start')
    img1 = cv2.imread(sys.argv[1], 0)
    img2 = cv2.imread(sys.argv[2], 0)
   
    u, v = solve(img1, img2)

    import matplotlib.pyplot as plt
    plt.quiver(u, v, u**2 + v**2)
    plt.show(1)
