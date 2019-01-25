import numpy as np

def specAct(clf, sample):
    ans = clf.activate(sample)
    if ans == 1:
        return 1.0
    return -1.0

class AdaBoostClassifier:
    def __init__(self, classifiers, clfWeights):
        self.classifiers = classifiers
        self.clfWeights = clfWeights

    def activate(self, sample):
        sm = 0
        for a, h in zip(self.clfWeights, self.classifiers):
            sm += a * specAct(h, sample)
        if sm > 0:
            return 1
        return 0


def buildAdaBoost(mkClf, data, labels, nStages):
    nSamples = data.shape[0]
    nFeats = data.shape[1]
    weights = np.ones(nSamples)/nSamples

    classifiers = []
    clfWeights  = []

    for stage in xrange(nStages):
        clf = mkClf(data, weights, labels)
        err = 0.0
        for x, w, l in zip(data, weights, labels):
            ans = clf.activate(x)
            if ans != l:
                err += w
        if err > 0.5:
            break
        alpha = 0.5 * np.log((1 - err)/err)
        classifiers.append(clf)
        clfWeights.append(alpha)
        for i in xrange(nSamples):
            h = specAct(clf, data[i])
            y = 1 if labels[i] == 1 else -1
            weights[i] = weights[i] * np.exp(-alpha * y * h)
        Z = sum(weights)
        print '>', stage, Z, err
        weights = weights / Z
    return AdaBoostClassifier(classifiers, clfWeights)

if __name__ == '__main__':
    pass
