import sys

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from random import random

if __name__ == '__main__':
    samples = [
        [1, 10],
        [12, 6],
        [7, 5],
        [0, 3],
        [45, 39],
        [6, 15],
        [100, 98],
        [21,8],
        [6, 9],
        [87, 85],
        [45, 20],
        [45, 2],
        [100, 101],
        [690, 699],
        [505, 600]
    ]
    samples = map(lambda x: map(float, x), samples)
    lables = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0]

    for i in xrange(10000):
        x = random() * 100
        y = random() * 100
        if i < 9000:
            x += 300
            y += 300
        samples.append([x, y])
        samples.append([y, x])
        res = (1 if abs(x - y) < 10 else 0)
        lables.append(res)
        lables.append(res)
        #print samples[-1], abs(x - y), lables[-1]

    print len(samples), len(lables)
    clf = svm.SVC()
    clf = clf.fit(samples, lables)
    for i in xrange(10):
        x = random() * 100
        y = random() * 100
        print x, y, abs(x - y), clf.predict([x, y])
    print clf.predict([11, 9])
    print clf.predict([1, 9])
    print clf.predict([1, 3])
    print clf.predict([3, 3])
    print clf.predict([30, 3])
    print clf.predict([300, 303])
