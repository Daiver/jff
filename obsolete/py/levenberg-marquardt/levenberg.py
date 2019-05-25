import numpy as np
import matplotlib.pyplot as plt

def lmIter(y, f_x, J, clambda=0):
    #A = -J
    #B = -(f_x - y)
    #return np.linalg.lstsq(A, B)[0]
    JtJ = np.dot(J.T, J)

    return np.linalg.solve(JtJ + clambda * np.diag(JtJ), np.dot(J.T, y - f_x))

    invJtJ = np.linalg.pinv(JtJ + clambda * np.diag(JtJ))
    return np.dot(invJtJ, np.dot(J.T, y - f_x))

#def func(x, w):
    #return w[0] * np.cos(x * w[1]) + w[1] * np.sin(w[0] * x)

#def grad(x, w):
    #da = np.cos(w[1] * x) + x * w[1] * np.cos(x * w[0])
    #db = -w[0] * x * np.sin(w[1] * x) + np.sin(x * w[0])
    #return np.array([da, db])

def func(x, w):
    return w[0] * np.cos(x * w[1])

def grad(x, w):
    da = np.cos(w[1] * x) 
    db = -w[0] * x * np.sin(w[1] * x) 
    return np.array([da, db])

def lm(xs, ys, initialW, nIter):
    J = np.zeros((len(xs), len(initialW)), dtype=np.float32)
    w = np.array(initialW)
    #print ys
    for iter in xrange(nIter):
        f_x = np.array(map(lambda x: func(x, w), xs))
        print "err", np.linalg.norm(f_x - ys)
        for i in xrange(len(J)) :
            g = grad(xs[i], w)
            J[i] = g
        #print J
        w += 1.0 * lmIter(ys, f_x, J, 0) 
        print 'w', w

    return w

if __name__ == '__main__':

    xs = np.linspace(0.0, 4.0, 1000)
    trueW = np.array([10.0, 12.0])
    ys = np.array(map(lambda x: func(x, trueW), xs), dtype=np.float32) + np.random.rand((len(xs)))

    newW = lm(xs, ys, [4.05, 12.44], 20)
    ys2 = np.array(map(lambda x: func(x, newW), xs))

    plt.plot(xs, ys, color="green")
    plt.plot(xs, ys2, color="red")
    plt.show()
