import numpy as np
import scipy.optimize

if __name__ == '__main__':
    c = np.ones(10)
    print c
    A = np.zeros((14 + 10, 10), dtype=np.float32)
    A[0, 0] = 1
    A[0, 5] = 1

    A[1, 0] = 1
    A[1, 1] = 1

    A[2, 0] = 1
    A[2, 4] = 1

    A[3, 1] = 1
    A[3, 6] = 1

    A[4, 1] = 1
    A[4, 2] = 1

    A[5, 2] = 1
    A[5, 7] = 1

    A[6, 2] = 1
    A[6, 3] = 1

    A[7, 3] = 1
    A[7, 8] = 1

    A[8, 3] = 1
    A[8, 4] = 1

    A[9, 4] = 1
    A[9, 9] = 1

    A[10, 9] = 1
    A[10, 8] = 1

    A[11, 9] = 1
    A[11, 5] = 1

    A[12, 5] = 1
    A[12, 6] = 1

    A[13, 6] = 1
    A[13, 7] = 1

    #A[14, 7] = 1
    #A[14, 8] = 1

    for i in xrange(10):
        A[14 + i, i] = 1
        #A[25 + i, i] = -1

    ub = np.ones(14 + 10)
    #ub *= 0.5
    ub[15:25] = 0
    #ub[25:] = -1

    A *= -1
    ub *= -1

    print scipy.optimize.linprog(c=c, A_ub=A, b_ub=ub)
