import cv2
import struct
import sys
import numpy as np
import matplotlib.pyplot as plt

def plotArrows(u, v):
    assert u.shape == v.shape
    u = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(u)))
    v = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(v)))
    angles = cv2.phase(u, v)
    #print angles
    '''angles = np.array([
            [0.5, 0.1, 0.3],
            [0.5, 0.1, 0.3],
            [1.5, 1.1, 1.3]
        ])'''
    #plt.arrow( 0.5, 0.8, 0.0, -0.2, fc="k", ec="k", head_width=0.05, head_length=0.1 )
    plt.xlim([0, angles.shape[0] + 1])
    plt.ylim([angles.shape[1] + 1, 0])
    #plt.arrow( 0.9, 0.1, 0.0, 0.2, fc="k", ec="k", head_width=0.05, head_length=0.1 )
    for i in xrange(angles.shape[0]):
        for j in xrange(angles.shape[1]):
            ni = 1.0 * np.cos(angles[i, j])
            nj = 1.0 * np.sin(angles[i, j])
            plt.arrow( i, j, ni, nj, fc="k", ec="k", head_width=0.05, head_length=0.1 )

    plt.show()

def drawArrows(v, u):
    '''
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.float32) 
    print hsv.shape
    mag, ang = cv2.cartToPolar(v, u)
    print mag.shape
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    print np.min(rgb), np.max(rgb)
    return rgb
    '''
    u1, v1 = map(cv2.pyrDown, [u, v])
    
    res = np.zeros((u.shape[0] *2, u.shape[1]*2))
    for i in xrange(0, u1.shape[0], 5):
        for j in xrange(0, u1.shape[1], 5):
            if (u1[i, j]**2 + v1[i, j]**2) < 0.2:
                continue
            cv2.line(res, (j*3, i*3), 
                    (int(j*3 + 2*v1[i, j]), int(i*3 + 2*u1[i, j])), 255)
    return res

def norm(img):
    min = np.min(img)
    max = np.max(img)
    print min, max
    img -= min
    img /= (max - min)
    img *= 255
    return img

if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        ff = f.read(4)
        w = struct.unpack('i', f.read(4))
        h = struct.unpack('i', f.read(4))
        print ff, w, h
        w, h = w[0], h[0]
        res = np.zeros((w, h), dtype=np.float)
        res2 = np.zeros((w, h), dtype=np.float)
        for i in xrange(w):
            for j in xrange(h):
                d = struct.unpack('f', f.read(4))[0]
                res[i, j] = d
                d = struct.unpack('f', f.read(4))[0]
                res2[i, j] = d
        print np.max(res), np.min(res)
        #res = norm(res)
        #res = res.astype(np.uint)
        #print res
        #cv2.imwrite('1.bmp', res)
        #res = None

        '''for i in xrange(w):
            for j in xrange(h):
                d = struct.unpack('f', f.read(4))[0]
                res2[i, j] = d'''
        #res2 = norm(res2)
        #res2 = res2.astype(np.uint)
        #print res2
        #cv2.imwrite('2.bmp', res2)

        '''img3 = cv2.phase(res, res2)
        img4 = cv2.magnitude(res, res2)
        ones = np.ones((w, h), dtype=np.float)
        result = cv2.merge([img3, ones, img4])
        #result = cv2.resize(result, (300, 300))
        result = cv2.cvtColor(result.astype(np.float32), cv2.COLOR_HSV2BGR)'''
        #cv2.imshow('', result)
        #cv2.waitKey()
        #print img3

        #print result.dtype
        #cv2.imshow('', result)
        #cv2.imwrite('3.bmp', result)
        #cv2.waitKey()
        #res = np.transpose(res)
        #res2 = np.transpose(res2)
        print res
        print res2
        res = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(res)))
        res2 = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(res2)))
        #plt.gca().invert_yaxis()
        #plt.gca().invert_xaxis()
        #res2 *= -1
        plt.quiver(res, res2, res**2 + res2**2)
        plt.show()
        #plotArrows(res2, res)
        #im = drawArrows(res, res2)
        #cv2.imshow('', im)
        #while 1:
        #    cv2.waitKey()
        '''plt.gca().invert_yaxis()
        x = np.linspace(-2,2,h)
        y = np.linspace(-2,2,w)
        plt.streamplot(x, y, res, res2, 5)
        plt.show()
        '''
