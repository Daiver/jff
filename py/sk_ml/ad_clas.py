import sys

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def getData(fname):
    lables = []
    samples = []
    with open(fname) as f:
        for line in f:
            sample = []
            tokens = line.replace(',', ' , ').split()
            lables.append(1 if tokens[-1] == 'ad.' else 0)
            samples.append([int(token) if token.isdigit() else 0 for token in tokens if token != ','])
    return lables, samples

def test(samples, lables, cross_value, clf):
    clf = clf.fit(samples[:-cross_value], lables[:-cross_value])
    count = 0
    for i in xrange(cross_value):
        if clf.predict(samples[-i])[0] != lables[-i]:
            count += 1
    print count

if __name__ == '__main__':
    fname = sys.argv[1]
    cross_value = 1000
    lables, samples = getData(fname)
    clf = RandomForestClassifier(n_estimators=1000)
    test(samples, lables,cross_value,clf)
    clf = svm.SVC()
    test(samples, lables, cross_value, clf)

