import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def drawProblem(points, a, b):
    ys = [a*x + b for x in points[:, 0]]
    plt.plot(points[:, 0], points[:, 1], 'ro', points[:, 0], ys)
    plt.show()

def lossFuncT(errSq):
    if errSq < 1.0:
        return errSq * (2.0 - errSq)
    else :
        return 1.0

def lossFuncTDer(errSq):
    if errSq < 1.0:
        return (2.0 - 2*errSq)
    else :
        return 0.0

def costFunc(X, Y, a, b):
    res = 0.0
    for x, y in zip(X, Y):
        res += lossFuncT((a*x + b - y)**2)
    return res

def costFuncDer(X, Y, a, b):
    res = np.array([0, 0])
    for x, y in zip(X, Y):
        res[0] += lossFuncTDer((a*x + b - y)**2) * 2 * (a*x + b - y) * x
        res[1] += lossFuncTDer((a*x + b - y)**2) * 2 * (a*x + b - y) 
    return res

if __name__ == '__main__':
    points = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [5, 5],
        [6, 6],
        #[5, 2.5],
        [4, 2],
        [3, 1.5],
        [2, 1.0],
        [1, 0.5],
        [-1, -0.5]
        ], dtype=np.float32)

    func = lambda vars: costFunc(points[:, 0], points[:, 1], vars[0], vars[1])
    grad = lambda vars: costFuncDer(points[:, 0], points[:, 1], vars[0], vars[1])

    res = minimize(func, [0, 0.1])
    print res

    drawProblem(points, res['x'][0], res['x'][1])
