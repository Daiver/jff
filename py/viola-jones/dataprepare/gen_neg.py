import cv2
import os
import sys
import random

srcDir    = sys.argv[1]
dstDir    = sys.argv[2]
nImgs     = int(sys.argv[3])
nPatches  = int(sys.argv[4])

def cutPatch(img, rect):
    x1, y1 = rect[0:2]
    x2, y2 = x1 + rect[2], y1 + rect[3]
    return img[y1:y2, x1:x2]

imgNames = filter(lambda x: x[-4:] == '.png' or x[-4:] == '.jpg', os.listdir(srcDir))
#print imgNames
size = (92, 112)
for iter in xrange(nImgs):
    index = random.randint(0, len(imgNames) - 1)
    img = cv2.imread(os.path.join(srcDir, imgNames[index]))
    #cv2.imshow('', img)
    for pInd in xrange(nPatches):
        xInd  = random.randint(0, img.shape[1] - size[0])
        yInd  = random.randint(0, img.shape[0] - size[1])
        patch = cutPatch(img, [xInd, yInd, size[0], size[1]])
        path = os.path.join(dstDir, '%s_%d_%d.png' % (imgNames[index], iter, pInd))
        cv2.imwrite(path, patch)
        #cv2.imshow('p', patch)
        #print img.shape
        #cv2.waitKey()
    #cv2.destroyAllWindows()
