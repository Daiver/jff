import sys
import cv2
import numpy as np
import numpy.linalg as lin
import reader

def takePixel(img, i, j):
    i = i if i >= 0 else 0
    j = j if j >= 0 else 0

    i = i if i < img.shape[0] else img.shape[0] - 1
    j = j if j < img.shape[1] else img.shape[1] - 1

    return img[i, j]

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

def vizMag(u, v):
    res = v**2 + u**2
    res2 = np.ones_like(u)
    res3 = cv2.phase(u, v)
    res = cv2.merge([res, res2, res3])
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)

    res = cv2.normalize(res, 0, 255, cv2.NORM_MINMAX)
    while res.shape[0] < 200:
        res = cv2.pyrUp(res)

    cv2.imshow('', res)
    cv2.waitKey(10)


def average(img):
    return cv2.filter2D(img.astype(np.float32), 5, kernel)

def findFlow(img1, img2, u=None, v=None):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    Idx = xDer(img1, img2)
    Idy = yDer(img1, img2)
    Idt = tDer(img1, img2)

    alpha = 3.9
    if u == None:
        u = np.zeros_like(img1)
    if v == None:
        v = np.zeros_like(img1)
    ua = average(u)
    va = average(v)

    print 'In iteration'

    b = 1.0 / (alpha**2 + Idx**2 + Idy**2)
    Idxb = Idx*b
    Idyb = Idy*b

    for k in xrange(10000):
        uo = u.copy()
        vo = v.copy()

        u = ua - Idxb * (Idx * ua + Idy * va + Idt) #/(alpha**2 + Idx**2 + Idy**2)
        v = va - Idyb * (Idx * ua + Idy * va + Idt) #/(alpha**2 + Idx**2 + Idy**2)
        ua = average(u)
        va = average(v)

        if np.allclose(u, uo) and np.allclose(v, vo):
            print 'end in', k
            break
        if k % 200 == 0:
            erru = sum(sum(abs(uo - u)))/uo.size
            errv = sum(sum(abs(vo - v)))/uo.size
            print k, 'errU', erru, 'errV', errv
            if errv + erru < 0.001:
                break

    return u, v

def takePixelExtrapolated(img, x, y):
    i, j = int(x), int(y)
    dx, dy = x - i, y - j
    p1 = takePixel(img, i, j)
    p2 = takePixel(img, i+1, j)
    p3 = takePixel(img, i, j+1)
    p4 = takePixel(img, i+1, j+1)
    return (p1 * (1 - dx) * (1 - dy) +
           p2 * (dx) * (1 - dy) +
           p3 * (1 - dx) * ( dy) +
           p4 * (dx) * ( dy))

def translate(img, u, v):
    print 'translate', u.shape, v.shape
    res = np.zeros_like(img)
    for i in xrange(res.shape[0]):
        for j in xrange(res.shape[1]):
            res[i, j] = takePixelExtrapolated(img, i + u[i, j], j + v[i, j])
    '''for i in xrange(res.shape[0]):
        for j in xrange(res.shape[1]):
            res[i, j] = takePixelExtrapolated(img, i - u[i, j], j - v[i, j])'''
    return  res

def translate2(img, u, v):
    print 'translate', u.shape, v.shape
    res = np.zeros_like(img)
    for i in xrange(res.shape[0]):
        for j in xrange(res.shape[1]):
            res[i, j] = takePixelExtrapolated(img, i - v[i, j], j - u[i, j])
    return  res


if __name__ == '__main__':
    img1 = cv2.imread(sys.argv[1], 0)
    img2 = cv2.imread(sys.argv[2], 0)

    #while img1.shape[0] > 1000:
    #    img1 = cv2.pyrDown(img1)
    #    img2 = cv2.pyrDown(img2)
    
    '''m1, m2 = [img1], [img2]
    while img1.shape[0] > 10:
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        m1.append(img1)
        m2.append(img2)'''

    '''img1 = np.array([
            [0,0,0, 0],
            [0,0,255, 0],
            [0,0,0, 0],
            [0,0,0, 0],
        ])
    img2 = np.array([
            [0,0,0, 0],
            [0,0,0, 0],
            [0,0,0, 0],
            [255,0,0, 0],
        ])'''

    #img1 = np.zeros((10,10))
    #img2 = np.zeros((10,10))
    #img1[0,0] = 5
    #img2[3,3] = 5


    #print img1
    #print img2

    '''u, v = None, None
    for im1, im2 in reversed(zip(m1, m2)):
        if u != None:
            u = cv2.resize(u, im1.shape)
            v = cv2.resize(v, im1.shape)
            print u.shape, v.shape, im1.shape, img2.shape
        u, v = findFlow(im1, im2, u, v)
        '''

    u, v = findFlow(img1, img2)
    #flow = cv2.calcOpticalFlowFarneback(img1, img2, 
    #                    0.5, 5, 2, 10, 3, 0.7, cv2.OPTFLOW_USE_INITIAL_FLOW)
    #print flow.shape
    #u, v = cv2.split(flow)

    #u, v = reader.readFlow(sys.argv[3])
    #u = np.transpose(u)
    #v = np.transpose(v)

    print u, v

    imgc = cv2.imread(sys.argv[1])
    #imgc = cv2.resize(imgc, (img1.shape[1], img1.shape[0]))
    res = translate(imgc, u, v)
    cv2.imwrite('tmp2.png', res)
    cv2.imshow('', res)
    cv2.waitKey()

    u = cv2.pyrDown(cv2.pyrDown(u))
    v = cv2.pyrDown(cv2.pyrDown(v))
    
    v *= -1
    import matplotlib.pyplot as plt
    plt.gca().invert_yaxis()
    #ax = plt.gca()
    #ax.set_ylim(ax.get_ylim()[::-1])

    plt.quiver(u, v, u**2 + v**2)
    plt.show(1)#'''


