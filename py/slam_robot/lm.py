import numpy as np

def numJac(func, dx=0.001):
    def inner(vars):
        vars = np.copy(vars)
        initial = func(vars)
        nVars = len(vars)
        nResiduals = len(initial)
        res = np.zeros((nResiduals, nVars), dtype=np.float64)
        for varInd in xrange(nVars):
            vars[varInd] += dx
            diff = (func(vars) - initial)/dx
            vars[varInd] -= dx
            res[:, varInd] = diff
        return res
    return inner

def gaussNewton(func, jac, initX, nIters, verbose=True):
    vars = np.copy(initX)
    for iter in xrange(nIters):
        J = jac(vars)
        residuals = func(vars)
        err = np.sum(residuals**2)
        print iter, 'err', err
        hess = J.T.dot(J) + np.eye(len(vars))*0.5
        #print hess
        delta = -np.linalg.lstsq(hess, J.T.dot(residuals))[0]
        vars += delta
    return vars

def levmar(func, jacFunc, initVars, tau, nIters, minGrad=0.00001, verbose=True):
    vars = np.copy(initVars)

    nu = 2.0

    residuals = func(vars)
    err       = residuals.dot(residuals)
    jac       = jacFunc(vars)
    jacT      = jac.T
    grad      = jacT.dot(residuals)
    hess      = jacT.dot(jac)
    mu        = tau
    gradNorm  = np.max(np.abs(grad))
    isFound   = gradNorm < minGrad

    for iter in xrange(nIters):
        if isFound:
            break
        if verbose:
            print iter, 'err', err, 'grad_inf', gradNorm, 'mu', mu
        delta = np.linalg.solve(hess + np.identity(len(vars)) * mu, -grad)
        varsNew = vars + delta
        newResiduals = func(varsNew)
        newErr = newResiduals.dot(newResiduals)
        predDiff = 0.5 * delta.dot(mu*delta - grad)
        realDiff = (err - newErr)
        gamma = realDiff/predDiff
        if gamma > 0.0:
            vars = varsNew
            residuals = newResiduals
            err = newErr
            jac       = jacFunc(vars)
            jacT      = jac.T
            grad      = jacT.dot(residuals)
            hess      = jacT.dot(jac)
            gradNorm  = np.max(np.abs(grad))
            isFound   = gradNorm < minGrad
            mu        = max(1.0/3, 1 - (2.0 * gamma)**3) * mu
            nu        = 2.0
        else:
            mu = mu * nu
            nu = 2.0 * nu
    return vars, err
