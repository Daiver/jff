import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#LinearRegression
#Ridge
#Lasso
#ElasticNet
#LassoLars


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def readData(fname):
    res = []
    with open(fname) as f:
        for s in f:
            if len(s) == 0 or s[0] == '#':
                continue

            res.append(map(float, s.split()[1:]))
    return np.array(res)

def visualizeWithPCA(data, values):
    pca = PCA(2)
    pca.fit(data)
    data_pca = pca.transform(data)
    for i in xrange(0, 20):

        clf = linear_model.LinearRegression()
        #clf = SVR(kernel='rbf', C=1e1, gamma=0.1, degree=5)
        #clf = RandomForestRegressor(n_estimators=1000, n_jobs=2)
        clf.fit(data_pca, values[:, i])
        xs = np.linspace(min(data_pca[:, 0]), max(data_pca[:, 0]))
        ys = np.linspace(min(data_pca[:, 1]), max(data_pca[:, 1]))
        prod = cartesian([xs, ys])
        zs = clf.predict(prod)
        print zs.shape, prod.shape

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(data_pca[:, 0], data_pca[:, 1], values[:, i], 'ro', label='data')

        ax.plot(zip(*prod)[0], zip(*prod)[1], zs, label='linear')
        ax.legend()
        plt.show()

def visualizeDim(data, values, dim1, dim2):
    for i in xrange(0, 5):
        clf = linear_model.LinearRegression()
        #clf = SVR(kernel='rbf', C=1e1, gamma=0.1, degree=5)
        #clf = RandomForestRegressor(n_estimators=1000, n_jobs=2)
        data_pca = np.array(zip(data[:, dim1], data[:, dim2]))
        clf.fit(data_pca, values[:, i])
        xs = np.linspace(min(data_pca[:, 0]), max(data_pca[:, 0]))
        ys = np.linspace(min(data_pca[:, 1]), max(data_pca[:, 1]))
        prod = cartesian([xs, ys])
        zs = clf.predict(prod)
        print zs.shape, prod.shape

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(data_pca[:, 0], data_pca[:, 1], values[:, i], 'ro', label='data')

        ax.plot(zip(*prod)[0], zip(*prod)[1], zs, label='linear')
        ax.legend()
        plt.show()


def testModel(data, values, clf):
    border = 100
    clf.fit(data[:border], values[:border])
    res = clf.predict(data[border:])
    err = sum((res - values[border:]) ** 2)/len(res[border:])
    print err, np.mean(res), np.mean(values)
    return err

if __name__ == '__main__':
    #metrixName  = '/home/daiver/3d/caesar-fitted-meshes-txt-csr-metrix.txt'
    #weightsName = '/home/daiver/3d/caesar-fitted-meshes-txt-csr-weights.txt'

    metrixName  = '/home/daiver/metric_statistic/caesar-nh-metric-1.txt'
    weightsName = '/home/daiver/metric_statistic/caesar-nh-weight-20.txt'

    data   = readData(metrixName)
    #data = normalize(data[:, :2])
    values = readData(weightsName)
    print 'Data shape', data.shape, 'Values shape', values.shape
    #values = normalize(values)
    #visualizeWithPCA(data, values)
    visualizeDim(data, values, 0, 1)
    exit(0)
#    for j in xrange(values.shape[1]):
        #for i in xrange(data.shape[1]):
            #print 'weight ind', j, 'attr ind', i
            #plt.plot(data[:, i], values[:, j], 'ro')
            #plt.show()
    

    #clf = linear_model.ElasticNetCV()
    #svr_rbf = SVR(kernel='linear', C=1e1, gamma=0.1)
#    svr_rbf = SVR(
            ##kernel='poly', 
            #kernel='linear', 
            ##kernel='rbf', 
            #C=1e1,
            #gamma=0.1,
            ##degree=15,
            ##max_iter=100000,
            #verbose=True)
    #clf = svr_rbf
    ferr1 = 0
    ferr2 = 0
    ferr3 = 0
    for i in xrange(values.shape[1]):
        clf1 = linear_model.LinearRegression()
        clf2 = RandomForestRegressor(n_estimators=100, n_jobs=2)
        clf3 = ExtraTreesRegressor(n_estimators=100, n_jobs=2)
        print i
        err1 = testModel(data, values[:, i], clf1)
        err2 = testModel(data, values[:, i], clf2)
        err3 = testModel(data, values[:, i], clf3)
        ferr1 += err1
        ferr2 += err2
        ferr3 += err3

    print ferr1, ferr2, ferr3

def main1():
    data   = readData('/home/daiver/3d/caesar-fitted-meshes-txt-csr-metrix.txt')
    #data = normalize(data[:, :2])
    values = readData('/home/daiver/3d/caesar-fitted-meshes-txt-csr-weights.txt')
    #values = normalize(values)
    for i in xrange(0, 10):
        for j1 in xrange(data.shape[1]):
            for j2 in xrange(data.shape[1]):
                if j1 == j2:
                    continue
                
                print i, j1, j2

                clf = linear_model.LinearRegression()
                #clf = SVR(kernel='rbf', C=1e1, gamma=0.1, degree=5)
                #clf = RandomForestRegressor(n_estimators=100, n_jobs=2)
                clf.fit(zip(data[:, j1], data[:, j2]), values[:, i])
                xs = np.linspace(min(data[:, j1]), max(data[:, j1]))
                ys = np.linspace(min(data[:, j2]), max(data[:, j2]))
                prod = cartesian([xs, ys])
                zs = clf.predict(prod)
                print zs.shape, prod.shape

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot(data[:, j1], data[:, j2], values[:, i], 'ro', label='data')
                ax.plot(zip(*prod)[0], zip(*prod)[1], zs, label='linear')
                ax.legend()
                plt.show()
    exit(0)
#    for j in xrange(values.shape[1]):
        #for i in xrange(data.shape[1]):
            #print 'weight ind', j, 'attr ind', i
            #plt.plot(data[:, i], values[:, j], 'ro')
            #plt.show()
    

    #clf = linear_model.ElasticNetCV()
    #svr_rbf = SVR(kernel='linear', C=1e1, gamma=0.1)
#    svr_rbf = SVR(
            ##kernel='poly', 
            #kernel='linear', 
            ##kernel='rbf', 
            #C=1e1,
            #gamma=0.1,
            ##degree=15,
            ##max_iter=100000,
            #verbose=True)
    #clf = svr_rbf
    ferr1 = 0
    ferr2 = 0
    ferr3 = 0
    for i in xrange(values.shape[1]):
        clf1 = linear_model.LinearRegression()
        clf2 = RandomForestRegressor(n_estimators=100, n_jobs=2)
        clf3 = ExtraTreesRegressor(n_estimators=100, n_jobs=2)
        print i
        err1 = testModel(data, values[:, i], clf1)
        err2 = testModel(data, values[:, i], clf2)
        err3 = testModel(data, values[:, i], clf3)
        ferr1 += err1
        ferr2 += err2
        ferr3 += err3

    print ferr1, ferr2, ferr3
