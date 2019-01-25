import operator
from fn import _

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def loadData(fname):
    labels,data = [],[]
    with open(fname) as f:
        for s in f:
            strs = s.lower().split()
            labels.append(strs[0])
            data.append(' '.join(strs[1:]))
    return labels, data


if __name__ == '__main__':
    fname = 'smallspam'
    fname = 'spamdata'
    raw_labels, raw_data = loadData(fname)
    #vectorizer = CountVectorizer()

    labels = [0 if x == 'spam' else 1 for x in raw_labels]

    tst_cnt = 10
    train_labels = labels[:-tst_cnt]
    test_labels  = labels[-tst_cnt:]
    train_data   = raw_data[:-tst_cnt]
    test_data    = raw_data[-tst_cnt]

    vectorizer = HashingVectorizer(n_features=5)
    data = vectorizer.fit_transform(train_data).toarray()
    #data = vectorizer.fit_transform(raw_data).toarray()
    print data.shape
    '''
    print 'Start PCA'
    pca = PCA(n_components=data.shape[1]/20)
    pca.fit(data)
    print 'Transform'
    data2 = map(lambda x: pca.transform(x).tolist()[0], data)
    '''
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(data, train_labels)
    #clf.fit(data, labels)
    tst = test_data #['HAVE A DATE ON SUNDAY WITH WILL!!', 'Txt the word: ']
    tst_data = vectorizer.transform(tst).toarray()
    print tst_data
    print clf.predict(tst_data)
    print test_labels
