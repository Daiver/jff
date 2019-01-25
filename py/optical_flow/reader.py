import cv2
import struct
import sys
import numpy as np
import matplotlib.pyplot as plt

def readFlow(fname):
    with open(fname, 'rb') as f:
        ff = f.read(4)
        w = struct.unpack('i', f.read(4))
        h = struct.unpack('i', f.read(4))
        w, h = w[0], h[0]
        res = np.zeros((h, w), dtype=np.float)
        res2 = np.zeros((h, w), dtype=np.float)
        for i in xrange(h):
            for j in xrange(w):
                d = struct.unpack('f', f.read(4))[0]
                res[i, j] = d
                d = struct.unpack('f', f.read(4))[0]
                res2[i, j] = d
    return res, res2

if __name__ == '__main__':
    res, res2 = readFlow(sys.argv[1])
    
    res = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(res)))
    res2 = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(res2)))
    #res2 *= -1
    plt.quiver(res, res2, res**2 + res2**2)
    plt.show()
