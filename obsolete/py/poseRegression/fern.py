import numpy as np
import random

import common

import FernBuiltins

#from inliner import inline

def getRangesFromData(data):
    return np.min(data, axis=0), np.max(data, axis=0)


class FernRegressor:
    def __init__(self, depth):
        self.depth  = depth

    def computeIndex(self, diffs):
        index = 0
        for i, x in enumerate(diffs):
            index *= 2
            if x >= 0:
                index += 1
        return index

    def activate(self, diffs):
        return self.bins[self.computeIndex(diffs)]

    def predict(self, diffs):
        diff = np.array(diffs)
        if len(diffs.shape) == 1:
            return [self.bins[self.computeIndex(diffs)]]
        else:
            return map(lambda x: self.predict(x)[0], diff)

    def evaluateError(self, diffs, values):
        err = 0.0
        for d, v in zip(diffs, values):
            err += abs(self.activate(d) - v)
        return err

    def fit(self, diffs, values):
        values = np.array(values, dtype=np.float32)
        self.bins       = np.zeros(int(np.exp2(self.depth)), dtype=np.float32)
        counts          = np.zeros(int(np.exp2(self.depth)), dtype=np.float32)
        mu = np.average(values)
        values = values - mu

        for v, x in zip(values, diffs):
            index = self.computeIndex(x)
            self.bins[index] += v
            counts[index] += 1

        beta = 0.01
        eps = 0.0001
        for i, c in enumerate(counts):
            self.bins[i] /= max([c + beta*len(diffs), eps]) 
        self.bins += mu

    def __repr__(self):
        return "Fern(%s)" % str(self.bins)

class RandomFernRegressor(FernRegressor):
    def __init__(self, depth):
        self.depth  = depth
        self.featuresIndices = []
    #@inline
    def computeIndex(self, diffs):
        index = 0
        for fid in self.featuresIndices:
            index <<= 1
            if diffs[fid] > 0:
                index += 1
        return index

    def activate(self, diff):
        return self.bins[self.computeIndex(diff)]

    def predict(self, diffs):
        diffs = np.array(diffs)
        if len(diffs.shape) == 1:
            lDiffs = diffs.shape[0]
            diffs = diffs.reshape((1, lDiffs))
#        if len(diffs.shape) == 1:
            #return [self.bins[self.computeIndex(diff)]]
            ##diffs = [diffs]
        #res = []
        #for d in diffs:
            #res.append(self.bins[self.computeIndex(d)])
        #return res
        return np.array(map(self.activate, diffs), dtype=np.float32)

    def fit(self, diffs, values):
#        diffs                = np.array(diffs, dtype=np.float32)
        #values               = np.array(values, dtype=np.float32)
        #self.bins            = np.zeros(int(np.exp2(self.depth)), dtype=np.float32)
        #counts               = np.zeros(int(np.exp2(self.depth)), dtype=np.float32)
        values = values.astype(np.float32)
        mu                   = np.average(values)
        values -= mu
        self.featuresIndices = [random.randrange(diffs.shape[1]) for _ in xrange(self.depth)]

        #for i in xrange(len(values)):
            #x = diffs[i]
            #v = values[i]
            #index = self.computeIndex(x)
            #self.bins[index] += v
            ##print index, v
            #counts[index] += 1

        self.bins = FernBuiltins.fitFernSimple(
                diffs, values, np.array(self.featuresIndices, dtype=np.int32), self.depth)

#        beta = 0.01
        #eps = 0.0001
        #for i, c in enumerate(counts):
            #self.bins[i] /= max([c + beta*len(diffs), eps]) 
        self.bins += mu

    def __repr__(self):
        return "RFern([%s] [%s])" % (str(self.bins), str(self.featuresIndices))


class FernRegressorBoosted:
    def __init__(self, depth, numberOfRepeats, numberOfFerns):
        self.ferns           = []
        self.depth           = depth
        self.numberOfFerns   = numberOfFerns
        self.numberOfRepeats = numberOfRepeats

    def __repr__(self):
        return ('FernRegressorBoosted(' + 
                '\n'.join(map(str, self.ferns)) + ')'
                )

    def fit(self, data, values):
        data   = np.array(data, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        for outerIter in xrange(self.numberOfFerns):
            bestError    = 1e10
            bestFern     = None

            for innerIter in xrange(self.numberOfRepeats):
                curFern = RandomFernRegressor(self.depth)
                #print 'Fit'
                curFern.fit(data, values)
                #print 'Predict'
                err = np.linalg.norm(values - curFern.predict(data))
                if err < bestError:
                    bestError = err
                    bestFern  = curFern

            #print values
            #print "PRED", bestFern.predict(data)
            values = values - bestFern.predict(data)
            #print values
            self.ferns.append(bestFern)

    def activate(self, data):
        res = 0
        for f in self.ferns:
            res += f.activate(data)
        return res

    def predict(self, data):
        data = np.array(data)
        if len(data.shape) == 1:
            data = [data]
        return np.array(map(self.activate, data), dtype=np.float32)


