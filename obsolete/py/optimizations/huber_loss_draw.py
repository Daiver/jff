import numpy as np
import scipy.optimize as opt

import matplotlib.pyplot as plt

if __name__ == '__main__':
    #obj = lambda x: x[0]*x[1]
    #cons1 = lambda x: 1 - (x[0]**2 + x[1]**2)
    #cons2 = lambda x: x[0]

    #print opt.fmin_cobyla(obj, [0.0, 0.1], [cons1, cons2])
    '''print opt.fmin_cobyla(obj, [0.0, 0.1], [
        lambda x:x[0] - 1,
        lambda x:x[1] - 1,
        lambda x:4 - x[0],
        lambda x:4 - x[1],
        ])'''

    '''
    print opt.fmin_cobyla(lambda x:-x[2], [1.0, 1.1, 1.1], [
            lambda x:x[0] - 1 - x[2],
            lambda x:x[1] - 1 - x[2],
            lambda x:4 - x[0] - x[2],
            lambda x:4 - x[1] - x[2],
            lambda x:x[2]
        ])
    '''
    '''print opt.fmin_cobyla(lambda x: x[1] + x[0], [1.0, 1.1], [
            lambda x: x[0],
            lambda x: x[1],
            lambda x: 2*x[0] + x[1] - 1,
            lambda x: x[0] + 3*x[1] - 1
        ])'''


    huber = lambda x: x**2 if abs(x) <= 1.3 else 1.3*(2*abs(x) - 1.3)
    X = np.arange(-2,2,0.1)
    Y = map(huber, X)

    plt.plot(X, Y, "r^")
    plt.show()
    #print opt.fmin_cobyla(lambda x: x[1] + x[0], [1.0, 1.1], [
    #            lambda x: x[0],
    #        ])
