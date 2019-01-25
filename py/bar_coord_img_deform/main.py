import numpy as np
import cv2
import sys

def barCentrForTriangle(p1, p2, p3, p):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x,  y  = p
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    det = 1.0/det
    l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3))*det
    l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3))*det
    l3 = 1.0 - l1 - l2
    return l1, l2, l3

def findTriangleForPoint(triangleIndices, vertices, p):
    for i, triInd in enumerate(triangleIndices):
        p1 = vertices[triInd[0]]
        p2 = vertices[triInd[1]]
        p3 = vertices[triInd[2]]
        l1, l2, l3 = barCentrForTriangle(p1, p2, p3, p)
        if (abs(abs(l1 + l2 + l3) - 1.0) < 0.00001) and l1 >= 0.0 and l2 >= 0.0 and l3 >= 0:
            return i, l1, l2, l3
    #print 'Strange'
    return -1, 0, 0, 0

def deform(img, triangleIndices, vertices1, vertices2):
    res = np.zeros(img.shape)
    for row in xrange(img.shape[0]):
        for col in xrange(img.shape[1]):
            ind, l1, l2, l3 = findTriangleForPoint(
                    triangleIndices, vertices2, np.array([row, col]))
            if ind == -1:
                continue
            #if ind in [1, 2, 0]:
                #continue
            triInd = triangleIndices[ind]
            p1 = vertices1[triInd[0]]
            p2 = vertices1[triInd[1]]
            p3 = vertices1[triInd[2]]
            p = p1*l1 + p2*l2 + p3*l3
            if p[0] < 0 or p[1] < 0 or p[0] >= img.shape[1] or p[1] >= img.shape[0]:
                continue
            #pix = img[int(p[1]), int(p[0])]
            #res[row, col] = pix
            pix = img[int(p[0]), int(p[1])]
            res[row, col] = pix
    return res

def drawTriangles(img, triangleIndices, vertices):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for triInd in triangleIndices:
        p1 = vertices[triInd[0]]
        p2 = vertices[triInd[1]]
        p3 = vertices[triInd[2]]
        cv2.line(
                img, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), (255, 0, 0))
        cv2.line(
                img, (int(p2[1]), int(p2[0])), (int(p3[1]), int(p3[0])), (255, 0, 0))
        cv2.line(
                img, (int(p1[1]), int(p1[0])), (int(p3[1]), int(p3[0])), (255, 0, 0))
    return img

def main():
    #img = np.arange(0, 4096 * 4).reshape((128, 128))/64
    img = cv2.imread('/home/daiver/Documents/nonsmile2.png', 0)
    
    triangleIndices = [
            [0, 1, 4],
            [1, 4, 3],
            [3, 4, 2],
            [2, 4, 0]
            ]
    vertices1 = np.array([
        [0, 0],
        [1 * img.shape[1], 0],
        [0, 1 * img.shape[0]],
        [1 * img.shape[1], 1 * img.shape[0]],
        [0.6 * img.shape[1], 0.5 * img.shape[0]]
        ], dtype=np.float32)

    vertices2 = np.array([
        [-20, 0],
        [1 * img.shape[1], 0],
        [-20, 1 * img.shape[0]],
        [1 * img.shape[1], 1 * img.shape[0]],
        [0.8 * img.shape[1], 0.5 * img.shape[0]]
        ], dtype=np.float32)


    img2 = deform(img, triangleIndices, vertices1, vertices2)
    img = deform(img, triangleIndices, vertices1, vertices1)

    diff = np.linalg.norm(img - img2)
    print diff

    img1t = drawTriangles(img.astype(np.uint8), triangleIndices, vertices1)
    img2t = drawTriangles(img2.astype(np.uint8), triangleIndices, vertices2)

    cv2.imshow('orig', img.astype(np.uint8))
    cv2.imshow('', img2.astype(np.uint8))
    #cv2.imshow('t1', img2t)
    #cv2.imshow('t2', img1t)
    cv2.waitKey()
    cv2.imwrite('orig2.png', img.astype(np.uint8))
    #cv2.imwrite('t12.png', img1t.astype(np.uint8))
    #cv2.imwrite('t22.png', img2t.astype(np.uint8))
    #cv2.imwrite('t2.png', img2t.astype(np.uint8))
    cv2.imwrite('r2.png', img2.astype(np.uint8))

if __name__ == '__main__':
    main()
