import cv2
import numpy as np
import sys

def nabla(I):
    h, w = I.shape
    G = np.zeros((h, w, 2), I.dtype)
    G[:, :-1, 0] -= I[:, :-1]
    G[:, :-1, 0] += I[:, 1:]
    G[:-1, :, 1] -= I[:-1]
    G[:-1, :, 1] += I[1:]
    return G

def nablaT(G):
    h, w = G.shape[:2]
    I = np.zeros((h, w), G.dtype)
    # note that we just reversed left and right sides
    # of each line to obtain the transposed operator
    I[:, :-1] -= G[:, :-1, 0]
    I[:, 1: ] += G[:, :-1, 0]
    I[:-1]    -= G[:-1, :, 1]
    I[1: ]    += G[:-1, :, 1]
    return I

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

def div(field):
    res = np.zeros((field.shape[0], field.shape[1]), dtype=np.float32)

    for i in xrange(1, field.shape[0] - 1):
        for j in xrange(1, field.shape[1] - 1):
            res[i, j] = field[i, j, 0] - field[i - 1, j, 0] + field[i, j, 1] - field[i, j - 1, 1]

    return res

def projectionOntoDisc(var):
    res = np.zeros_like(var)
    for i in xrange(var.shape[0]):
        for j in xrange(var.shape[1]):
            norm = max(1.0, np.linalg.norm(var[i, j]))
            res[i, j] = var[i, j] / norm
    return res

def projectionOntoDiscFast(var):
    norm = np.fmax(1.0, np.linalg.norm(var, axis=2))
    res = var / norm[:, :, np.newaxis]
    return res
    
def projectionOntoDiscOneDimension(var):
    norm = np.fmax(1.0, np.abs(var))
    res = var / norm
    return res

def cost(u, img, clambda):
    gx, gy = cv2.split(nabla(u))
    return np.linalg.norm(clambda*(u - img)) + np.linalg.norm(gx) + np.linalg.norm(gy)

def tvl1Denoising(img, clambda, omega, tau):
    cv2.imshow('', (normalize(img)))
    cv2.waitKey(10)
    xi0  = np.zeros((img.shape[0], img.shape[1], 2))
    phi0 = np.zeros((img.shape[0], img.shape[1]))
    u0   = np.copy(img) #np.zeros((img.shape[0], img.shape[1]))
    uN   = u0

    for iter in xrange(221):
        phi = projectionOntoDiscOneDimension(phi0 + omega * clambda * (uN - img))
        xi  = projectionOntoDiscFast(xi0 + omega*nabla(uN))
        u   = u0 + tau*(-nablaT(xi0)) - tau*clambda*phi0
        uN  = u + 0.9*(u - u0)
    
        if iter % 10 == 0:
            print 'Iter', iter, cost(u, img, clambda), np.linalg.norm(u - u0)
            cv2.imshow('', (normalize(u)))
            cv2.waitKey(10)

        u0   = u
        phi0 = phi
        xi0  = xi

    return u0

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 0).astype(np.float32)
    cv2.imshow('1', (normalize(img)))
    u = tvl1Denoising(img, 1.0, 0.16, 0.9)
    print 'Res', np.linalg.norm(u - img)
    cv2.imwrite('res.png', normalize(u))
    cv2.imshow('', (normalize(u)))
    cv2.waitKey()
