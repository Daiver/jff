import sys

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def sample_from_line(line):
    tmp = line.lower().replace('.', ' . ').replace(',', ' , ').replace('(', ' ( ').replace(')', ' ) ').replace('!', ' ! ').replace('/', ' / ').split()
    if len(tmp) < 2:
        return ('', [])
    return (1 if tmp[0] == 'ham' else 0, tmp[1:])

def make_dct(fname):
    dct = {}
    i = 0
    with open(fname) as f:
        for line in f:
            for word in sample_from_line(line)[1]:
                if word.isdigit(): word = '<digit>'
                if word not in dct:
                    dct[word] = i
                    i += 1
    return dct

def freq_dct(sample):
    res = {}
    for x in sample:
        if x not in res: res[x] = 0
        res[x] += 1
    return res

def make_wordbag(dct, sample):
    tmp = freq_dct(sample[1])
    return [1 if key in tmp else 0 for key in dct]

def make_data_from_file(fname):
    all_words = make_dct(fname)
    samples = []
    lables = []
    with open(fname) as f:
        for line in f:
            tmp = sample_from_line(line)
            samples.append(make_wordbag(all_words, tmp))
            lables.append(tmp[0])
    return (lables, samples)

def test(samples, lables, cross_value, clf):
    clf = clf.fit(samples[:-cross_value], lables[:-cross_value])
    count = 0
    for i in xrange(cross_value):
        if clf.predict(samples[-i])[0] != lables[-i]:
            count += 1
    print count

if __name__ == '__main__':
    cross_value = 3000
    n_estimators = 30
    print "cross_value %d n_estimators %d" % (cross_value, n_estimators)
    lables, samples = make_data_from_file(sys.argv[1])
    print 'readed %d' % len(samples)
    clf = RandomForestClassifier(n_estimators=n_estimators)
    test(samples, lables, cross_value, clf)
    clf = svm.SVC()
    test(samples, lables, cross_value, clf)
    #print clf
    #print dir(clf)
#
