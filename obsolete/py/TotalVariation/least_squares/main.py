import numpy as np
import cv2
import sys

def normalize(v):
    #norm=np.max(v)
    #return v/norm
    img = np.copy(v)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if img[i, j] < 0:
                img[i, j] = 0
            if img[i, j] > 255:
                img[i, j] = 255
    return img.astype(np.uint8) 

def getNeighs(img, i, j):
    res = []
    if i > 0:
        res.append(img[i - 1, j])
    if j > 0:
        res.append(img[i, j - 1])
    if i < img.shape[0] - 1:
        res.append(img[i + 1, j])
    if j < img.shape[1] - 1:
        res.append(img[i, j + 1])
    return res

def denoising(img, clambda):
    x0 = np.copy(img)
    #x0 = np.zeros(img.shape)

    nk = 4

    for iterCounter in xrange(0, 500):
        xNext = np.zeros(img.shape)
        for i in xrange(img.shape[0]):
            for j in xrange(img.shape[1]):
                neighs = getNeighs(x0, i, j)
                a = (1 + clambda*len(neighs))
                b = img[i, j]
                others = -sum([clambda * x for x in neighs])
                xNext[i, j] = 1.0/a * (b - others)
        print iterCounter, np.linalg.norm(xNext - x0), np.linalg.norm(xNext - img)
        cv2.imshow('New', normalize(xNext))
        cv2.waitKey(1)
        x0 = xNext

    return xNext

if __name__ == '__main__':
    imgOrig    = cv2.imread(sys.argv[1])
    imgOrig = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('orig.png', imgOrig)
    cv2.imshow('Orig', imgOrig)
    imgDenoised = denoising(imgOrig, 3.4)

    imgDenoised = normalize(imgDenoised)
    cv2.imwrite('tmp.png', imgDenoised)

    cv2.imshow('New', imgDenoised)
    cv2.waitKey()
