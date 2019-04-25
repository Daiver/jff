from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

import json

def loadData(fname):
    labels, samples = [], []
    with open(fname) as f:
        l = -1
        n = ''
        s = ''
        for line in f:
            if len(line) < 3: continue
            i1 = line.find('"')
            i2 = line.find('"', i1 + 1)
            i3 = line.find('"', i2 + 1)
            name = line[i2 + 1:i3-1]
            #line2 = line[:i1] + line[i2 + 1:]
            label = int(line[:i1 - 1])
            sam = json.loads(line[i3:]).lower()
            if name != n:
                if n != '':
                    labels.append(l)
                    samples.append(s)
                n = name
                l = label
                s = sam
            else:
                sam += ' ' + s
            '''.replace(
                    ',', ' , ').replace('.', ' . ').replace(
                    '(', ' ').replace( ')', ' ')'''
        #samples.append(s)
        #labels.append(l)
    return labels, samples

if __name__ == '__main__':
    labels, raw_data = loadData('tweets_data')
    res = set()
    for x in raw_data:
        s = x.split()
        for y in s: res |= {y}
    print len(res)
    #labels = labels[:62000]
    #raw_data = raw_data[:62000]
    print len(raw_data)
    tst_cnt = 20
    train_labels   = labels[:-tst_cnt]
    test_labels    = np.array(labels[-tst_cnt:])
    train_raw_data = raw_data[:-tst_cnt]
    test_data      = raw_data[-tst_cnt:]
    #vectorizer = CountVectorizer()
    print 'Vecs'
    vectorizer = HashingVectorizer(n_features=10000)
    train_data = vectorizer.fit_transform(train_raw_data).toarray()
    print train_data.shape
    test_data = vectorizer.transform(test_data).toarray()

    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)
    y_pred = gnb.predict(test_data)
    print y_pred.shape
    print test_data.shape
    print test_labels.shape
    print y_pred
    print test_labels
    print("Number of mislabeled points : %d" % (test_labels != y_pred).sum())

    #clf = RandomForestClassifier(n_estimators=10)
    #clf.fit(train_data, train_labels)
    #clf.predict(test_data)
    #print test_labels
