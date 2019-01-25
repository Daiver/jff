import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def lossFunc(err, weight, tauSq):
    errSq = err**2
    weightSq = weight**2
    return 2 * weightSq * errSq / tauSq + (1.0 - weightSq)**2

def lossFuncMin(err, tauSq):
    func = lambda w: lossFunc(err, w[0], tauSq)
    w = minimize(func, [0.01])['x'][0]
    #print w
    return lossFunc(err, w, tauSq)

if __name__ == '__main__':
    xs = np.linspace(-1.5, 1.5)
    ys1 = [lossFunc(x, 1, 1) for x in xs]
    ys2 = [lossFunc(x, 2, 1) for x in xs]
    ys3 = [lossFunc(x, 0.5, 1) for x in xs]
    ysn = [lossFuncMin(x, 1) for x in xs]
    plt.plot(xs, ys1, xs, ys2, xs, ys3, xs, ysn)
    plt.show()
