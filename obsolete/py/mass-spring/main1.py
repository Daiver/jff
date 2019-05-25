import numpy as np
from matplotlib import pyplot as plt

dt = 0.001

def simulate(x, y, vx, vy):
    g  = -9.8
    vy += g*dt
    x += vx*dt
    y += vy*dt
    return x, y, vx, vy

if __name__ == '__main__':
    x = 0
    y = 10
    vx = 1
    vy = 0
    t = 0
    xs = []
    ys = []
    while y > 0.0:
        x, y, vx, vy = simulate(x, y, vx, vy)
        print 'x:',x, 'y:',y, 'vx:',vx, 'vy:',vy
        t += dt
        xs.append(x)
        ys.append(y)
    print t
    txs = []
    tys = []
    for i in np.linspace(0, t, 10000):
        txs.append(i*vx)
        tys.append(10 - 9.8*(i**2)/2.0)

    plt.plot(xs, ys, '-', txs, tys, '--')
    plt.show()

