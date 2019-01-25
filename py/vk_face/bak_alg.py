import gmpy
import mpmath
import scipy
import numpy as np
import cv2

def hyperF(a, b, c, z):
    res = 1
    product_item = a * b * z / c
    res += product_item
    for k in xrange(2, 100):
        product_item *= (a + k - 1) * (b + k - 1) * z /((c + k - 1) * k)
        res += product_item
    return res
        
#test for hyperF
def test_hyper_F():
    #slow implementation of hyperF
    def testF(a, b, c, z):
        res = 1
        for k in xrange(1, 100):
            product_item = 1
            for l in xrange(k):
                product_item *= (a + l) * (b + l) / ((1 + l) * (c + l))
            product_item *= z**k
            res += product_item
        return res

    import random
    for i in xrange(100):
        a, b, c, z = random.random() * 10, random.random() * 10, 0.1 + random.random() * 10, random.random() * 10
        h = hyperF(a, b, c, z)
        t = testF(a, b, c, z)
        print a, b, c, z, h, t, h - t

def kraw(p, q, N, n, x):
    #return (q**n) * gmpy.comb(x, n) * hyperF(-n, x - N, x - n, -p/q)
    return (q**n) * gmpy.comb(x, n) * hyperF(-n, -x, -N, -p/q)

def ro(N, n, p):
    return gmpy.comb(N, n) * (p ** n) * ((1 - p) ** n)

def weight_j(p, q, N, x):
    return gmpy.comb(N, x) * (p ** x) * (q**(N - x))

print ro(10, 5, 0.5), ro(10, 6, 0.5)
res = 0
N = 100
n = 5
p = -1
q = 0.5
for x in xrange(N+1):
    res += weight_j(p, q, N, x) * (kraw(p, q, N, n, x) ** 2)
print res, ro(N, n, p)
