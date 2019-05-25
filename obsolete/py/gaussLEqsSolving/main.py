import numpy as np

def solve(a, b, init_x=None):
    assert a.shape[0] == b.shape[0] and a.shape[1] == b.shape[0]
    if init_x == None:
        init_x = np.zeros_like(b)
    x = init_x.copy()
    while True:
        for i in xrange(x.shape[0]):
            aii = a[i, i]
            sm = 0
            for j in xrange(0, x.shape[0]):
                if i != j:
                    sm += a[i, j] * x[j] 
            x[i] = 1.0/aii * (b[i] - sm)
        if sum(sum(abs(x - init_x))) < 0.000001:
            break
        init_x = x.copy()
    return x


if __name__ == '__main__':
    print solve(
            np.array([
                    [16, 3.],
                    [7, -11]
                ]),
            np.transpose(np.array([[11,13.]]))
        )
