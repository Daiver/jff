import alg
import numpy as np
import cv2
import os
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

'''
data = [[1, 1], [1, 0], [0, 0], [0.1, 0]]
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
print kmeans
kmeans.fit(data)
print kmeans
print kmeans.predict([0, 0.2])
'''

if __name__ == '__main__':
    dirs = map(lambda x: '/home/daiver/Downloads/orl_faces/s%d' % x, range(1, 10))
    data = []
    print 'reading '
    for dr in dirs:
        for fname in os.listdir(dr):
            data.append(alg.desc(cv2.resize(cv2.imread(os.path.join(dr,fname), 0), (50, 50))))

    kmeans = KMeans(init='k-means++', n_clusters=9, n_init=10)
    print kmeans
    kmeans.fit(data)
    print kmeans
    for i, dr in enumerate(dirs):
        print i, dr
        for fname in os.listdir(dr):
            desc = alg.desc(cv2.resize(cv2.imread(os.path.join(dr,fname), 0), (50, 50)))
            print i, kmeans.predict(desc)
