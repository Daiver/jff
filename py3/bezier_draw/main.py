import matplotlib.pyplot as plt
import numpy as np

def binominal(n, k):
    #only for cubic now
    assert n == 3 and k >= 0 and k <= n
    return [1, 3, 3, 1][k]


def curveCoordAtT(controlPoints, t):
    n = 3
    assert len(controlPoints) == 4
    assert t >= 0 and t <= 1
    res = np.zeros(2)
    for i in range(n + 1):
        res += binominal(n, i) * ((1.0 - t)**(n - i)) * (t ** i) * controlPoints[i]
    return res

if __name__ == '__main__':
    controlPoints = np.array([
        [0, 0],
        [0.2, 1],
        [0.8, -1],
        [1, 0]
    ])

    ts = np.linspace(0, 1)

    ps = np.array([curveCoordAtT(controlPoints, t) for t in ts])

    plt.plot(controlPoints[:, 0], controlPoints[:, 1], 'ro')
    plt.plot(ps[:, 0], ps[:, 1])
    plt.show()

