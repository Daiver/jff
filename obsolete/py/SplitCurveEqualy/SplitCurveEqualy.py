import numpy as np

def mkPiesewiseLinearCurveLengthApproximation(xs, ys):
    nPoints = len(xs)
    assert len(ys) == nPoints
    res = np.zeros((nPoints), dtype=np.float64)
    for i in xrange(1, nPoints):
        x  = xs[i]
        xp = xs[i - 1]
        y  = ys[i]
        yp = ys[i - 1]
        length = np.sqrt((x - xp)**2 + (y - yp)**2)
        res[i] = length

    return res

def equallySplitCurveLengthApproximation(xs, ys, lengthApprox, nDesirePoints):
    assert nDesirePoints > 0
    nPoints = len(xs)
    assert len(ys) == nPoints
    assert len(lengthApprox) == nPoints
    assert nPoints > 1
    finalLength = sum(lengthApprox)
    segmentLength = finalLength / (nDesirePoints - 1)

    walkedDistance = 0.0
    res = [(xs[0], ys[0])]
    for i in xrange(1, nPoints):
        curLen = lengthApprox[i]
        walkedDistance += curLen
        while walkedDistance - segmentLength >= -1e-6:
            xp, yp = xs[i - 1], ys[i - 1]
            x , y  = xs[i]    , ys[i]
            dx, dy = x - xp, y - yp
            distanceOverlap = walkedDistance - segmentLength
            lineCoeff = 1.0 - float(distanceOverlap) / float(curLen)
            resX = xp + dx * lineCoeff
            resY = yp + dy * lineCoeff
            res.append((resX, resY))
            walkedDistance -= segmentLength

    assert len(res) == nDesirePoints
    return np.array(res)

def splitFunctionOnInterval(
        curveFunction, startInterval, finishInterval, nApproxSegements, nDesirePoints):
    f = curveFunction
    xs = np.linspace(startInterval, finishInterval, nApproxSegements)
    ys = f(xs)
    lengthApprox = mkPiesewiseLinearCurveLengthApproximation(xs, ys)
    points = equallySplitCurveLengthApproximation(xs, ys, lengthApprox, nDesirePoints)
    return points

