import cv2
import os
import sys

srcDir = sys.argv[1]
dstDir = sys.argv[2]

dirs = []
for x in os.listdir(srcDir):
    absDir = os.path.join(srcDir, x)
    if os.path.isdir(absDir):
        dirs.append(x)

for d in dirs:
    for fname in os.listdir(os.path.join(srcDir, d)):
        fullName = os.path.join(srcDir, d, fname)
        img = cv2.imread(fullName)
        cv2.imwrite(os.path.join(dstDir, "%s_%s.png" % (d, fname)), img)
