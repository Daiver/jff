import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

data = [[1, 1], [1, 0], [0, 0], [0.1, 0]]
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
print kmeans
kmeans.fit(data)
print kmeans
print kmeans.predict([0, 0.2])
