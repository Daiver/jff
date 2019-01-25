import numpy as np

def proxGradStep(func, grad, proxOperator, beta, stepSize, x):
    #fix it later
    funcVal = func(x)
    gradVal = grad(x)
    for _ in xrange(30):
        gradStep = x - gradVal * stepSize
        z = proxOperator(gradStep, stepSize)
        #predictedFunc = func(z) + 1.0/2/stepSize * np.sum(np.square(z - x))
        predictedFunc = funcVal + gradVal.dot(z - x) + 1.0/2.0/stepSize * np.sum(np.square(z - x))
        if predictedFunc >= func(z):
            break
        stepSize *= beta
    return z, stepSize

def proximalGradient(func, grad, proxOperator, beta, stepSize, x, verbose = False):
    n_iters = 60
    for iter in xrange(n_iters):
        x1, stepSize1 = proxGradStep(func, grad, proxOperator, beta, stepSize, x)
        diff = x1 - x
        dx = np.linalg.norm(diff)**2
        x = x1
        if stepSize1 < stepSize:
            stepSize = stepSize1
        else:
            pass
            #stepSize /= beta
        if verbose:
            print iter, func(x), stepSize, stepSize1, dx
        if dx < 1e-6:
            #print iter
            break
    return x

def accProxGradStep(func, grad, proxOperator, beta, stepSize, x, y):
    #fix it later
    funcVal = func(x)
    gradVal = grad(x)
    for _ in xrange(30):
        gradStep = y - grad(y) * stepSize
        z = proxOperator(gradStep, stepSize)
        #predictedFunc = func(z) + 1.0/2/stepSize * np.sum(np.square(z - x))
        predictedFunc = funcVal + gradVal.dot(z - x) + 1.0/2.0/stepSize * np.sum(np.square(z - x))
        if predictedFunc >= func(z):
            break
        stepSize *= beta
    return z, stepSize

def acceleratedProximalGradient(func, grad, proxOperator, beta, stepSize, x, verbose = False):
    n_iters = 50
    diff = np.zeros(x.shape)
    for iter in xrange(n_iters):
        w = float(iter) / (iter + 3)
        y = x + diff * w
        x1, stepSize1 = accProxGradStep(func, grad, proxOperator, beta, stepSize, x, y)
        diff = x1 - x
        dx = np.linalg.norm(diff)
        x = x1
        if stepSize1 < stepSize:
            stepSize = stepSize1
        else:
            stepSize /= beta
        if verbose:
            print iter, func(x), stepSize, stepSize1, dx
        if dx < 1e-6:
            #print iter
            break
    return x

