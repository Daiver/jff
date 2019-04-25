import numpy as np

def giniImp(freqs):
    return 1.0 - sum(freqs**2)

criteria = giniImp

def infGain(freqsAll, freqsL, wSumL, freqsR, wSumR):
    #nFreqsAll = nFreqsL + nFreqsR
    return (criteria(freqsAll) 
            - wSumL * criteria(freqsL) 
            - wSumR * criteria(freqsR))
            #- nFreqsL/float(nFreqsAll) * criteria(freqsL) 
            #- nFreqsR/float(nFreqsAll) * criteria(freqsR))

def computeFreqsWeighted(labels, weights, nClasses):
    res = np.zeros(nClasses)
    #print 'W', weights
    sm = sum(weights)
    normalizedWeights = weights/sm
    for i, x in enumerate(labels):
        res[x] += normalizedWeights[i]
        #res[x] += weights[i]
    #if len(labels) > 0:
        #return res/len(labels)
    return res

def divideWeighted(data, weights, labels, attrInd, attrVal):
    dataL, dataR, weightsL, weightsR, labelsL, labelsR = [], [], [], [], [], []
    for x, w, l in zip(data, weights, labels):
        if x[attrInd] >= attrVal:
            dataR.append(x)
            weightsR.append(w)
            labelsR.append(l)
        else:
            dataL.append(x)
            weightsL.append(w)
            labelsL.append(l)
    return (
            np.array(dataL),
            np.array(weightsL),
            np.array(labelsL),
            np.array(dataR),
            np.array(weightsR),
            np.array(labelsR)
            )

def findBestDvdWeighted(data, weights, labels, nClasses):
    nFeats = data.shape[1]
    bestGain = 0
    freqsAll = computeFreqsWeighted(labels, weights, nClasses)

    for j in xrange(nFeats):
        for i in xrange(data.shape[0]):
            _, weightsL, labelsL, _, weightsR, labelsR = (
                        divideWeighted(data, weights, labels, j, data[i, j]) )
            freqsL = computeFreqsWeighted(labelsL, weightsL, nClasses)
            freqsR = computeFreqsWeighted(labelsR, weightsR, nClasses)
            gain = infGain(freqsAll, freqsL, sum(weightsL), freqsR, sum(weightsR))
            #print gain
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

def makeStumpWeighted(data, weights, labels, nClasses):
    gain, attr, val = findBestDvdWeighted(data, weights, labels, nClasses)
    _, weightsL, labelsL, _, weightsR, labelsR = divideWeighted(data, weights, labels, attr, val)
    freqsL = computeFreqsWeighted(labelsL, weightsL, nClasses)
    freqsR = computeFreqsWeighted(labelsR, weightsR, nClasses)
    return DecisionStump(attr, val, freqsL, freqsR)

if __name__ == '__main__':
    pass
