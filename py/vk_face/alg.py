import gmpy
import mpmath
import scipy
import numpy as np
import cv2
import openturns

def fact(n):
    res = 1
    for i in xrange(2, n):
        res *= i
    return res

def weight(x, p, N):
    return mpmath.binomial(N, x) * (p ** x) * ((1 - p)**(N - x))

def kraw(n, x, p, N):
    return openturns.KrawtchoukFactory(N, p).build(n)(x)
    #return mpmath.hyp2f1(-n, -x, -N, 1./p, accurate_small=False)
    #return mpmath.hyp2f1(-n, -x, -N, 1./p, maxprec=1000)

def K_weight(n, x, p, N):
    return openturns.KrawtchoukFactory(N, p).build(n)(x)
    return kraw(n, x, p, N) * mpmath.sqrt(weight(x, p, N)/ro(n, p, N))

def ro(n, p, N):
    #((-1) ** n) * (((1-p)/p) ** n) * (fact(n)/)
    return (1 if n % 2 == 0 else -1) * (((1-p)/p) ** n) * (1./mpmath.binomial(-N, n))
    #return ((-1) ** n) * (((1-p)/p) ** n) * (1./mpmath.binomial(-N, n))
    '''return ((-1) ** n) * (((1-p)/p) ** n) * (1./mpmath.binomial(-N, n))
    print ((-1) ** n) * (((1-p)/p) ** n) * (1./mpmath.binomial(-N, n))
    print ((-1) ** n) * (((1-p)/p) ** n)* (mpmath.fac(n)/mpmath.qp(-N, n=n)) #* (1./mpmath.binomial(-N, n))
    return ((-1) ** n) * (((1-p)/p) ** n)* (mpmath.fac(n)/mpmath.qp(-N, n=n)) #* (1./mpmath.binomial(-N, n))
    '''

def Q(arr, p1, p2, n, m):
    N = arr.shape[0]
    M = arr.shape[1]
    sum = 0
    for x in xrange(N):
        for y in xrange(M):
            sum += K_weight(n, x, p1, N - 1) * K_weight(m, y, p2, M - 1) * arr[x, y]
    return sum

def Q_inv(shape, q, p1, p2, n, m):
    N = shape[0]
    M = shape[1]
    sum = 0
    for x in xrange(N):
        for y in xrange(M):
            sum += q * K_weight(n, x, p1, N - 1) * K_weight(m, y, p2, M - 1) 
    return sum

def tst1():
    N = 50
    p = 0.5
    for n in xrange(0, N + 1):
        m = n
        res1 = sum(
                    weight(x, p, N) * kraw(n, x, p, N) * kraw(m, x, p, N)
                    for x in xrange(0, N + 1)
                )
        print res1, ro(n, p, N)
        res1 = sum(
                    K_weight(n, x, p, N)
                    for x in xrange(0, N + 1)
                )
        #print res1, ro(n, p, N)

def desc(arr, p, q):
    #p = 0.3
    #q = 0.2
    return [
        Q(arr, p, q, 0, 0),
        Q(arr, p, q, 1, 0),
        Q(arr, p, q, 0, 1),
        Q(arr, p, q, 1, 1),
        Q(arr, p, q, 2, 0),
        Q(arr, p, q, 0, 2),
        Q(arr, p, q, 2, 2),
        Q(arr, p, q, 3, 0),
        Q(arr, p, q, 0, 3),
        Q(arr, p, q, 1, 3),
        Q(arr, p, q, 3, 1),
        Q(arr, p, q, 2, 3),
        Q(arr, p, q, 3, 2),
        Q(arr, p, q, 3, 3),
        Q(arr, p, q, 4, 0),
        Q(arr, p, q, 0, 4),
        Q(arr, p, q, 4, 1),
        Q(arr, p, q, 1, 4),
        Q(arr, p, q, 4, 2),
        Q(arr, p, q, 2, 4),
        Q(arr, p, q, 4, 3),
        Q(arr, p, q, 3, 4),
        Q(arr, p, q, 4, 4),
    ]

def tst2(arr):
    p = 0.3
    q = 0.3
    print Q(arr, p, q, 0, 0)
    print Q(arr, p, q, 1, 0)
    print Q(arr, p, q, 0, 1)
    print Q(arr, p, q, 1, 1)
    print Q(arr, p, q, 2, 0)
    print Q(arr, p, q, 0, 2)
    print Q(arr, p, q, 2, 2)
    '''for i in xrange(arr.shape[0]):
        for j in xrange(arr.shape[1]):
            print i, j, Q_inv(arr.shape, q, 0.2, 0.3, i, j)'''

def tst3(): #if __name__ == '__main__':
    arr = np.array([[0,1,0],[1,1,1],[0,1,1]])
    arr = cv2.resize(cv2.imread('/home/daiver/Downloads/1.png', 0), (50, 50))
    arr2 = cv2.resize(cv2.imread('/home/daiver/Downloads/2.png', 0), (50, 50))
    arr3 = cv2.resize(cv2.imread('/home/daiver/Downloads/3.png', 0), (50, 50))
    print '1'
    tst2(arr)
    print '2'
    tst2(arr2)
    print '3'
    tst2(arr3)

def tst4():
    import ChebyshevDesc
    TD = ChebyshevDesc.ChebyshevDesc((50, 50))

tst4()
