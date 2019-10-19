import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    # path_to_src = "/home/daiver/Frame00000.jpg"
    path_to_src = "/home/daiver/Frame00064.jpg"

    # path_to_dst = "/home/daiver/Frame00006.jpg"
    # path_to_dst = "/home/daiver/Frame00057.jpg"
    # path_to_dst = "/home/daiver/Frame00061.jpg"
    path_to_dst = "/home/daiver/Frame00067.jpg"

    img1 = cv2.imread(path_to_src, 0)  # queryImage
    img2 = cv2.imread(path_to_dst, 0)  # trainImage

    img1 = cv2.pyrDown(img1)
    img2 = cv2.pyrDown(img2)

    # Initiate SIFT detector
    # orb = cv2.ORB_create()
    # orb = cv2.KAZE_create()
    # orb = cv2.AKAZE_create(threshold=0.001)
    orb = cv2.AKAZE_create(threshold=0.0000001)
    # orb = cv2.MSER_create()
    # orb = cv2.BRISK_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    print(f"len(kp1) = {len(kp1)} len(kp2) = {len(kp2)}")

    # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], flags=2, outImg=None)
    img4 = cv2.drawMatches(img1, kp1, img2, kp2, matches[50:100], flags=2, outImg=None)
    img5 = cv2.drawMatches(img1, kp1, img2, kp2, matches[100:150], flags=2, outImg=None)
    img6 = cv2.drawMatches(img1, kp1, img2, kp2, matches[150:200], flags=2, outImg=None)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[50:100], flags=2, outImg=None)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[100:150], flags=2, outImg=None)
    cv2.imshow("0-50", img3)
    cv2.imshow("50-100", img4)
    cv2.imshow("100-150", img5)
    cv2.imshow("150-200", img6)
    cv2.waitKey()
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    # knn_matches = matcher.knnMatch(des1, des2, 2)
    # # -- Filter matches using the Lowe's ratio test
    # ratio_thresh = 0.7
    # good_matches = []
    # for m, n in knn_matches:
    #     if m.distance < ratio_thresh * n.distance:
    #         good_matches.append(m)
    # # -- Draw matches
    # img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    # cv2.drawMatches(img1, kp1, img2, kp2, good_matches, img_matches,
    #                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # # -- Show detected matches
    # cv2.imshow('Good Matches', img_matches)


if __name__ == '__main__':
    main()
