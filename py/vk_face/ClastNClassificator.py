import cv2
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

class ClastNClassificator:
    def __init__(self, N):
        self.N = N
        self.kmeans = KMeans(init='k-means++', n_clusters=N, n_init=10)
        self.clfs = [RandomForestClassifier(n_estimators=1000) for i in xrange(N)]
        #self.clfs = [cv2.createFisherFaceRecognizer() for i in xrange(N)]

    def fit(self, data, labels):
        self.kmeans.fit(data)
        splitted_data = [[] for i in xrange(self.N)]
        splitted_labels = [[] for i in xrange(self.N)]
        for i, x in enumerate(data):
            idx = self.kmeans.predict(x)[0]
            splitted_data[idx].append(x)
            splitted_labels[idx].append(labels[i])
        for clf, data, labels in zip(self.clfs, splitted_data, splitted_labels):
            clf.fit(data, labels)

    def predict(self, sample):
        return self.clfs[self.kmeans.predict(sample)[0]].predict(sample)
