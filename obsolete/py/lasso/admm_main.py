import numpy as np

#without line search
def admm(proxF, proxG, zInit, nIters, clambda):
    nVars = len(zInit)
    z = np.copy(zInit)
    u = np.zeros(nVars)
    zOld = z
    for iterInd in xrange(nIters):
        x = proxF(clambda, z - u)
        z = proxG(clambda, x + u)
        u = u + x - z
        diff = np.linalg.norm(zOld - z)
        zOld = z
        print iterInd, x, u, diff
    return x

def proxF(A, b):
    AtA  = A.T.dot(A)
    Atb  = A.T.dot(b)
    def inner(clambda, v):
        AtAI = np.eye(AtA.shape[0]) + AtA * clambda
        rhs  = -v + clambda * Atb
        return np.linalg.solve(AtAI, rhs)
        
    return inner

def proxG(regCoeff = 1):
    #intercept is last variable
    def inner(clambda, v):
        res = np.copy(v)
        clambda = clambda / regCoeff
        for i, x in enumerate(res):
            if i == len(res) - 1:
                continue
            if x >= clambda:
                res[i] = x - clambda
            elif np.abs(x) <= clambda:
                res[i] = 0.0
            else:
                res[i] = x + clambda
        return res
    return inner

if __name__ == '__main__':
    A = np.array([
        [1, 1],
        [2, 1],
        [3, 1]
        ])
    b = np.array([
        1, 2, 3
        ])
    proxF_inst = proxF(A, b)
    proxG_inst = proxG(0.01)
    x = np.zeros(2)
    res = admm(proxF_inst, proxG_inst, x, 40, 0.001)
    print res
    print A.dot(res)
