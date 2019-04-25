
import haar
import find_best_thr

class Stump:
    def __init__(self, featType, rect, threshold, polarity):
        self.featType = featType
        self.rect = rect
        self.threshold = threshold
        self.polarity = polarity

    def predict(self, integralImg):
        haarFunc = haar.haarFeaturesFuncs[self.featType]
        negR, posR = haarFunc(self.rect[0], self.rect[1], self.rect[2], self.rect[3])
        val = haar.computeHaarFeature(integralImg, negR, posR)
        if val * self.polarity > self.threshold * self.polarity:
            return 1
        else:
            return 0

def learnStump(integralImages, weights, labels, coordSteps, scaleSteps):
    height, width = integralImages[0].shape
    bestErr = 1.0e10
    clf = None
    for coordStepX in coordSteps:
        for coordStepY in coordSteps:
            for scaleStepX in scaleSteps:
                for scaleStepY in scaleSteps:
                    rect = [int(coordStepX * width), 
                            int(coordStepY * height), 
                            int(scaleStepX * width), 
                            int(scaleStepY * height)]

                    if rect[0] + rect[2] >= width:
                        continue
                    if rect[1] + rect[3] >= height:
                        continue
                    for featName, featFunc in haar.haarFeaturesFuncs.iteritems():
                        negR, posR = featFunc(rect[0], rect[1], rect[2], rect[3])
                        values = [haar.computeHaarFeature(x, negR, posR) for x in integralImages]
                        thr, err, pol = find_best_thr.findBestThr(values, weights, labels)
                        if err < bestErr:
                            bestErr = err
                            clf = Stump(featName, rect, thr, pol)
    return clf, bestErr
