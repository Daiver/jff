import dec_stump
import numpy as np
import cv2

def trainAdaBoostClassifier(integralImages, labels, coordSteps, scaleSteps, nClassifiers):
    nNeg = len(filter(lambda x: abs(x) < 0.0001, labels))
    nPos = labels.shape[0] - nNeg
    weights = np.zeros(labels.shape[0])
    for i in xrange(weights.shape[0]):
        if labels[i] == 1:
            weights[i] = 1.0/nPos
        else:
            weights[i] = 1.0/nNeg


    classifiers = []
    alphas      = []

    for iter in xrange(nClassifiers):
        weights = weights/sum(weights)
        clf, err = dec_stump.learnStump(integralImages, weights, labels, coordSteps, scaleSteps)
        if err > 0.5:
            break

        beta = err/(1.0 - err)
        alpha = np.log(1.0/beta)
        classifiers.append(clf)
        alphas.append(alpha)

        for i in xrange(weights.shape[0]):
            ans = clf.predict(integralImages[i])
            et = 0 if ans == labels[i] else 1
            weights[i] = weights[i] * np.power(beta, 1 - et)
        print iter, beta, alpha, err

    return classifiers, alphas

def predict(classifiers, alphas, sample):
    thr = 0.5 * sum(alphas)
    sm = 0.0
    for c, a in zip(classifiers, alphas):
        sm += a * c.predict(sample)
    if sm > thr:
        return 1
    return 0

if __name__ == '__main__':
    pass
