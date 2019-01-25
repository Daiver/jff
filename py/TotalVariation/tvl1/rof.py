import numpy as np
import cv2
import sys

import tvl1

def rofDenoising(img, clambda, omega, tau):
    img = img.astype(np.float32)
    u0  = np.copy(img)
    xi0 = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
    cv2.imshow('orig', tvl1.normalize(img))
    for iter in xrange(200):
        xi = tvl1.projectionOntoDiscFast(xi0 + omega*tvl1.nabla(u0))
        u  = (u0 + tau * (-tvl1.nablaT(xi)) + tau * clambda * img)/(1.0 + tau*clambda)

        if iter % 10 == 0:
            print iter, np.linalg.norm(u - u0)
            cv2.imshow('', tvl1.normalize(u))
            cv2.waitKey(10)

        xi0 = xi
        u0 = u

    return u0

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 0)
    u = rofDenoising(img, 1.0/12, 0.16, 0.9)
    print 'End'
    cv2.imshow('', tvl1.normalize(u))
    cv2.imwrite('res.png', tvl1.normalize(u))
    cv2.waitKey()
