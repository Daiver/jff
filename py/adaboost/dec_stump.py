import numpy as np

def giniImp(freqs):
    return 1.0 - sum(freqs**2)

criteria = giniImp

def infGain(freqsAll, freqsL, nFreqsL, freqsR, nFreqsR):
    nFreqsAll = nFreqsL + nFreqsR
    return (criteria(freqsAll) 
            - nFreqsL/float(nFreqsAll) * criteria(freqsL) 
            - nFreqsR/float(nFreqsAll) * criteria(freqsR))

def computeFreqs(labels, nClasses):
    res = np.zeros(nClasses)
    for x in labels:
        res[x] += 1
    if len(labels) > 0:
        return res/len(labels)
    return res

def divide(data, labels, attrInd, attrVal):
    dataL, dataR, labelsL, labelsR = [], [], [], []
    for x, l in zip(data, labels):
        if x[attrInd] >= attrVal:
            dataR.append(x)
            labelsR.append(l)
        else:
            dataL.append(x)
            labelsL.append(l)
    return (
            np.array(dataL),
            np.array(labelsL),
            np.array(dataR),
            np.array(labelsR)
            )

def findBestDvd(data, labels, nClasses):
    nFeats = data.shape[1]
    bestGain = 0
    freqsAll = computeFreqs(labels, nClasses)
    for j in xrange(nFeats):
        for i in xrange(data.shape[0]):
            _, labelsL, _, labelsR = divide(data, labels, j, data[i, j])
            freqsL = computeFreqs(labelsL, nClasses)
            freqsR = computeFreqs(labelsR, nClasses)
            gain = infGain(freqsAll, freqsL, len(labelsL), freqsR, len(labelsR))
            if gain > bestGain:
                bestGain = gain
                bestAttr = j
                bestVal  = data[i, j]
    return bestGain, bestAttr, bestVal

class DecisionStump:
    def __init__(self, attr, val, freqsL, freqsR):
        self.attr = attr
        self.val = val
        self.freqsL = freqsL
        self.freqsR = freqsR

    def activate(self, sample):
        res = self.freqsR if sample[self.attr] >= self.val else self.freqsL
        return np.argmax(res)

def makeStump(data, labels, nClasses):
    gain, attr, val = findBestDvd(data, labels, nClasses)
    _, labelsL, _, labelsR = divide(data, labels, attr, val)
    freqsL = computeFreqs(labelsL, nClasses)
    freqsR = computeFreqs(labelsR, nClasses)
    return DecisionStump(attr, val, freqsL, freqsR)

if __name__ == '__main__':
    pass
