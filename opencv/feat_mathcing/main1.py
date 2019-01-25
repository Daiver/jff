import cv2
import numpy as np
from matplotlib import pyplot as plt

imgDir = '../images/'

#img1 = cv2.imread(imgDir + 'box.png',0)          # queryImage
#img2 = cv2.imread(imgDir + 'box_in_scene.png',0) # trainImage

img1 = cv2.imread('/home/daiver/tmp/3.jpg', 0) # trainImage
img2 = cv2.imread('/home/daiver/tmp/4.jpg', 0) # queryImage

# Initiate SIFT detector
#orb = cv2.ORB_create(edgeThreshold=5, fastThreshold=5)
#orb = cv2.xfeatures2d.SIFT_create(edgeThreshold=5)
orb = cv2.xfeatures2d.SURF_create(hessianThreshold=5)
#orb = cv2.SIFT()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

print len(kp1), len(kp2)

bf = cv2.BFMatcher(cv2.NORM_L2SQR)
#bf = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=True)
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

print len(matches)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:5], flags=2, outImg=None)

plt.imshow(img3),plt.show()
