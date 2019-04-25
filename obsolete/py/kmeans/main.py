import numpy as np
from matplotlib import pyplot as plt
import random

def findNearestInd(point, points):
    #print points
    return np.argmin([np.linalg.norm(point - points[i, :]) for i in xrange(len(points))])

def center(points):
    res = np.zeros((points.shape[1]))
    for p in points: res += p
    res /= points.shape[0]
    return res

def kmeans(points, k):
    indices = range(len(points))
    random.shuffle(indices)
    centroids = np.array([points[i] for i in indices[:k]])
    labels = [0]*len(points)
    for iter in xrange(0, 10):
        labels = np.array([findNearestInd(points[i], centroids) for i in xrange(len(points))], dtype = np.uint)
        centroids = np.array([center(points[labels == i]) for i in xrange(k)])
    return labels

if __name__ == '__main__':
    points = np.array([
        [1, 1],
        [2, 2],
        [2, 1],
        [4, 4],
        [5, 5],
        [6, 5],
        [4, 5]
        ], dtype=np.float32)

    print kmeans(points, 2)
    #plt.plot(points[:, 0], points[:, 1], 'x', markersize=10.0)
    #plt.show()
