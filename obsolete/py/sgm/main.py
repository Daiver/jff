import numpy as np
import cvxpy

def trainLinearRegressor(featuresMat, targetDeltas):
    nSamples, nFeats = featuresMat.shape
    nOutputs = targetDeltas.shape[1]
    b = cvxpy.Variable(nOutputs)
    R = cvxpy.Variable(nOutputs, nFeats)
    residuals = featuresMat * R.T + cvxpy.kron(cvxpy.Constant(np.ones((nSamples))), b) - targetDeltas
    func = cvxpy.sum_squares(residuals)
    prob = cvxpy.Problem(cvxpy.Minimize(func))
    prob.solve(verbose=False)
    return R.value, b.value

def activate(regressor, featuresMat):
    R, b = regressor
    return R * (featuresMat) + b
    return R.dot(featuresMat) + b

if __name__ == '__main__':
    #func4Feat = np.exp
    #rngX = (-1, 5)
    func4Feat = np.square
    rngX = (1, 2)
    xs = np.linspace(rngX[0], rngX[1])
    ys = map(func4Feat, xs)
    targetX = 0.0
    nIters = 10

    xs = xs.reshape((-1, 1))
    xscur = xs
    regressors = []
    for iterInd in xrange(nIters):
        feats  = func4Feat(xscur)
        deltas = -(xscur - targetX)
        regressor = trainLinearRegressor(feats, deltas)
        regressors.append(regressor)
        regOut = activate(regressor, feats)
        xscur += regOut
        print 'iter', iterInd, np.sum(np.square(xscur))

    xstest = np.linspace(rngX[0], rngX[1], 200).reshape((-1, 1))
    xscur = xstest
    for i, regressor in enumerate(regressors):
        feats  = func4Feat(xscur)
        regOut = activate(regressor, feats)
        xscur += regOut
        print 'iter', i, np.sum(np.square(xscur))
