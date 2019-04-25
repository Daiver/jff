import numpy as np

import matplotlib.pyplot as plt

def hubert(x):
    k = 1.345
    if abs(x) <= k:
        return 1/2.0 * (x**2)/k
        #return 1/2.0 * (x**2)
    return abs(x) - 1.0/2*(k)
    #return k*abs(x) - 1.0/2*(k**2)

if __name__ == '__main__':
    x = np.arange(-6,6,0.01)
    y = map(hubert, x)
    y2 = map(lambda x: x**2, x)
    plt.plot(x, y, "r--")
    plt.show()
