import numpy as np
import cv2
from urllib.request import urlopen


import albumentations as albu


def download_image(url):
    data = urlopen(url).read()
    data = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def draw_points(image, points, radius=3):
    im = image.copy()
    for (x, y) in points:
        cv2.circle(im, (int(x), int(y)), radius, (0, 255, 0), -1)
    return im


def main():
    image = download_image('https://habrastorage.org/webt/_m/8j/vb/_m8jvb11szwten8kxx5a5rgkhcw.jpeg')
    points = [(100, 100), (720, 410), (1100, 400), (1700, 30),
              (300, 650), (1570, 590), (560, 800), (1300, 750),
              (900, 1000), (910, 780), (670, 670), (830, 670),
              (1000, 670), (1150, 670), (820, 900), (1000, 900)]

    points = np.array(points, dtype=np.float32)
    points /= 4
    image = cv2.pyrDown(cv2.pyrDown(image))

    cv2.imshow('original', draw_points(image, points))
    cv2.waitKey()

    augmentator = albu.Compose([albu.ShiftScaleRotate(p=1)], keypoint_params={'format': 'xy'})
    for i in range(5):
        augmentation_result = augmentator(image=image, keypoints=points)
        new_img = augmentation_result["image"]
        new_points = augmentation_result["keypoints"]
        cv2.imshow('augmented', draw_points(new_img, new_points))
        cv2.waitKey()


main()
