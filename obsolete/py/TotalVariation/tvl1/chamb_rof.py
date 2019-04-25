import numpy as np
import cv2
import sys

import tvl1

def rofDenoising(img, clambda, omega, tau):
    img = img.astype(np.float32)
    u0  = img #np.zeros_like(img) #np.copy(img)
    xi0 = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
    cv2.imshow('orig', tvl1.normalize(img))
    for iter in xrange(40):
        t = (tvl1.nabla(-tvl1.nablaT(xi0) - img/clambda))
        norm = (1.0 + tau * np.linalg.norm(t, axis=2))
        xi = ((xi0 + tau * t ) / norm[:,:,np.newaxis])
        u  = img + omega * (-tvl1.nablaT(xi))

        if iter % 1 == 0:
            print iter, np.linalg.norm(u - u0)
            cv2.imshow('', tvl1.normalize(u))
            cv2.waitKey(10)

        xi0 = xi
        u0 = u

    return u0

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 0)
    u = rofDenoising(img, 0.08, 0.9, 0.1)
    print 'End'
    cv2.imshow('', tvl1.normalize(u))
    cv2.imwrite('res.png', tvl1.normalize(u))
    cv2.waitKey()
