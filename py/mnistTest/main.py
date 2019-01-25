import os, cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import pickle

def bytes2int(b): 
    def fullBin(x): 
        b = bin(x)[2:]
        return '0' * (8 - len(b)) + b
    return int(''.join(map(lambda x: fullBin(ord(x)), b)), 2)

def readImages(fname):
    with open(fname, 'rb') as f:
        m, size, rows, cols = [bytes2int(f.read(4)) for i in xrange(4)]
        for k in xrange(size):
            img = np.zeros((rows, cols), dtype=np.uint8)
            for i in xrange(rows):
                for j in xrange(cols):
                    img[i, j] = ord(f.read(1))
            yield img

def readLabels(fname):
    with open(fname, 'rb') as f:
        m, size = [bytes2int(f.read(4)) for i in xrange(2)]
        for k in xrange(size):
            yield ord(f.read(1))

def descriptor(img):
    #cv2.imshow('', img) ; cv2.waitKey()
    #img = cv2.pyrDown(img)
    #cntr = list(cv2.Sobel(img, 5, 1, 1).reshape((-1)))
    return list(img.reshape((-1))) #+ list(cv2.pyrDown(img).reshape((-1)))
    #return list(img.reshape((-1))) + cntr + list(cv2.pyrDown(img).reshape((-1)))

if __name__ == '__main__':
    fnamei_train = '/home/daiver/Downloads/train-images-idx3-ubyte'
    fnamel_train = '/home/daiver/Downloads/train-labels-idx1-ubyte'
    fnamei_test = '/home/daiver/Downloads/t10k-images-idx3-ubyte'
    fnamel_test = '/home/daiver/Downloads/t10k-labels-idx1-ubyte'
    trainI, trainL = map(descriptor, readImages(fnamei_train)), list(readLabels(fnamel_train))
    testI, testL   = map(descriptor, readImages(fnamei_test)), list(readLabels(fnamel_test))

    clf = RandomForestClassifier(n_estimators=50)
    #clf = svm.SVR()
    #anova_filter = SelectKBest(f_regression, k=50)
    # 2) svm
    #clf = svm.SVC(kernel='linear', gamma=0.001)

    #anova_svm = Pipeline([('anova', anova_filter), ('svm', clf)])
    print 'Start learning'
    #clf.fit(trainI, trainL)
    clf.fit(trainI, trainL)
    pickle.dump(clf, open('classifier.model', 'w'))
    print 'Start test'
    errNum = 0
    for l, i in zip(testL, testI):
        ans = clf.predict(i)
        #ans = clf.predict(i)
        if ans[0] != l: 
            errNum += 1
            print ans, l
    print errNum
    print errNum/float(len(testL))
