#import alg
import itertools
import numpy as np
import cv2
import os
import KrawtchoukDesc
import ChebyshevDesc
from ClastNClassificator import ClastNClassificator
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

#size = (92, 112)
size = (100, 100)

KD = KrawtchoukDesc.KrawtchoukDesc(0.5, 0.5, size)
TD = ChebyshevDesc.ChebyshevDesc(size)
def desc(arr):
    #tmp = clf2.predict(arr)
    res = TD(arr)
    #return res
    return KD(arr) #+ res
    return (arr)
    return KD(arr) #+ [ tmp[1]]

def processImage(img):
    return cv2.resize(img, (size[1], size[0]))

def selectToTrain(seq):
    l = int(0.7 * len(seq))
    num = l if l > 1 else 1
    return seq[:num]

if __name__ == '__main__':
    min_fin_err = 200
    num_of_classes = 40
    dump_dir = '/home/daiver/dumps/vk_face'
    dirs = map(lambda x: os.path.join(dump_dir, x, 'faces'), os.listdir(dump_dir))
    labels = []
    images = []
    trainimg = []
    print 'reading '
    for i, dr in enumerate(dirs):
        print i, dr
        tmpI = []
        tmpL = []
        l = os.listdir(dr)
        l.sort()
        for j, fname in enumerate(l):
            img = processImage(cv2.imread(os.path.join(dr,fname), 0))
            tmpI.append(img)
            tmpL.append(i)
        labels.append(tmpL)
        images.append(tmpI)

    trainlabels = np.array(sum((selectToTrain(x) for x in labels), [])) #map(desc, trainimg)
    clf2 = cv2.createFisherFaceRecognizer()
    print 'Train fisher'
    clf2.train(sum((selectToTrain(x) for x in images), []), trainlabels)
    data = map(lambda x: map(desc, x), images)
    traindata = sum((selectToTrain(x) for x in data), []) #map(desc, trainimg)
    total_train = len(traindata)
    #clf = svm.SVC()
    clf = KMeans(init='k-means++', n_clusters=2, n_init=10)
    clf = RandomForestClassifier(n_estimators=1000)
    clf = ClastNClassificator(3)
    print 'Training'
    clf.fit(traindata, trainlabels)
    #clf.fit(traindata)
    print clf
    fin_errors = 0
    fin_errors2 = 0
    fin_errors3 = 0
    fin_errors4 = 0
    total = 0
    for ind, tmp in enumerate(zip(sum(labels, []), sum(data, []), sum(images, []))):
        l, x, i = tmp
        total += 1
        pred = clf.predict(x)
        pred2 = clf2.predict(i)
        if pred[0] != l:
            fin_errors += 1
        if pred2[0] != l:
            fin_errors2 += 1
        if pred[0] != l and pred2[0] != l:
            fin_errors3 += 1
        if (pred[0] if pred2[1] > 1000 else pred2[0]) != l:
            #print '>>>', pred2, l, pred
            fin_errors4 += 1
        print l, pred, pred2
        cv2.imshow('orig', sum(images,[])[ind])
        cv2.imshow('true', images[l][0])
        cv2.imshow('forest', images[pred[0]][0])
        cv2.imshow('cv', images[pred2[0]][0])
        #cv2.waitKey()
    print 'Krawtchouk && random forest', fin_errors, float(fin_errors)/(total - total_train)
    print 'opencv fisher', fin_errors2, float(fin_errors2)/(total - total_train)
    print 'cross error', fin_errors3
    print 'threshold', fin_errors4, total
    print total_train
