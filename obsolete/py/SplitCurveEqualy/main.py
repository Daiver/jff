import numpy as np
import matplotlib.pyplot as plt

from SplitCurveEqualy import *

if __name__ == '__main__':
    f = lambda x: np.ones(x.shape) * 5
    f = lambda x: x * 2 + 1
    f = lambda x: 0.5 * x**3 - 1.0 * x + 2.0

    nApproxSegements = 20000
    startInterval  = 0.0
    finishInterval = 2.0
    xs = np.linspace(startInterval, finishInterval, nApproxSegements)
    ys = f(xs)

    lengthApprox = mkPiesewiseLinearCurveLengthApproximation(xs, ys)
    nDesirePoints = 10
    points = equallySplitCurveLengthApproximation(xs, ys, lengthApprox, nDesirePoints)
    print lengthApprox
    print sum(lengthApprox)
    print points
    print len(points), '/', nDesirePoints
    print splitFunctionOnInterval(f, startInterval, finishInterval, nApproxSegements, nDesirePoints)

    distDiffs = np.linalg.norm(points[1:] - points[:-1], axis=1)
    print distDiffs
    print np.std(distDiffs)

    plt.plot(xs, ys, 'g-')
    plt.plot(points[:, 0], points[:, 1], 'go')
    xs2 = np.linspace(startInterval, finishInterval, 1e5)
    ys2 = f(xs2)
    plt.plot(xs2, ys2, 'r--')

    plt.show()

