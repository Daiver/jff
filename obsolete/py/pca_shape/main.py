import numpy as  np
from sklearn.decomposition import PCA

def readModel(fname):
    res = []
    with open(fname) as f:
        for s in f:
            res += map(float, s.split())
    return np.array(res)

def writeComponents(fname, components):
    with open(fname, 'w') as f:
        pass

data = np.array([
            (readModel('/home/daiver/bodies2/csr4001a.mat.txt')),
            (readModel('/home/daiver/bodies2/csr4002a.mat.txt')),
            (readModel('/home/daiver/bodies2/csr4003a.mat.txt')),
            (readModel('/home/daiver/bodies2/csr4004a.mat.txt')),
            (readModel('/home/daiver/bodies2/csr4005a.mat.txt')),
        ], dtype=np.float32)
        #], dtype=np.float64)



print data.shape
pca = PCA(6)
print pca.fit(data)

means = pca.mean_
components = pca.components_

trans = np.dot(components, (data[2] - means))
recovered = np.dot(components.T, trans) + means
print np.linalg.norm(recovered - data[2])
print pca.explained_variance_
print pca.explained_variance_ratio_

#trans = pca.transform(data)
#recovered = pca.inverse_transform(trans)

#print sum(sum(abs(recovered - data)))
#print np.linalg.norm(recovered - data)
