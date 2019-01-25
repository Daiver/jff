import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

import common

def readPoints(fname):
    res = []
    with open(fname) as f:
        for s in f:
            res.append( map(int, s.split()))
    return np.array(res)

if __name__ == '__main__':
    points = np.array([
                [0, 5],
                [5, 0],
                [5, 5],
                [0, 0]
            ])
    points = readPoints('./fail_vertexes_dump')
    #points = readPoints('../points/cropped1.txt')
    tri = Delaunay(points, qhull_options='QJ')
    triangles = tri.simplices.copy()
    for t in triangles:
        p = map(lambda x:points[x], t)
        for i, p0 in enumerate(points):
            if i not in t:
                if not common.checkDelaunay2(p, p0):
                    print 'ERROR!', 'i', i, 't', t, 'p', p,'p0', p0
    plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    plt.gca().invert_yaxis()
    plt.show()
