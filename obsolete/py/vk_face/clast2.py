import numpy as np
import cv2
import os
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import forest

'''
data = [[1, 1], [1, 0], [0, 0], [0.1, 0]]
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
print kmeans
kmeans.fit(data)
print kmeans
print kmeans.predict([0, 0.2])
'''

if __name__ == '__main__':
    path = '/home/daiver/dumps/vk_face/44017485/faces'
    images = map(lambda x: cv2.imread(x, 0), map(lambda x: os.path.join(path, x), os.listdir(path)))

    data = map(lambda x: forest.processImage(x).reshape((-1,)), images)

    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    print kmeans
    kmeans.fit(data)
    print kmeans
    for i, x in enumerate(images):
        print i
        desc = forest.processImage(x).reshape((-1,))
        print i, kmeans.predict(desc)
        cv2.imshow('', x)
        cv2.waitKey()
