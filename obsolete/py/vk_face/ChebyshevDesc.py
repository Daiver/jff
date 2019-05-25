import numpy as np
class ChebyshevPolynom:
    def __init__(self, p, N):
        self.p = p
        self.N = float(N)
        if self.p <= 1: 
            self.const1 = 1 - N
        else:
            self.const1 = float(2*p - 1)
            self.const2 = (p - 1) * (1 - ((p - 1)**2)/(N**2))
        
    def __call__(self, other, x):
        if self.p == 0: return 1
        if self.p == 1: 
            return (2*x - self.const1)/self.N
        return float(self.const1*(other[1](other, x))*(other[self.p - 1](other, x)) - self.const2*(other[self.p - 2](other, x)))/self.p


class ChebyshevDesc:
    def __init__(self, size):
        self.N, self.M = size
        N, M = size
        mp = 5
        mq = 5
        ksp = map(lambda x: ChebyshevPolynom(x, self.N), xrange(mp))
        ksq = map(lambda x: ChebyshevPolynom(x, self.M), xrange(mq))
        ros = self.getRos(mp)
        self.Tk = []
        steps = []
        for i in xrange(1, mp):
            for j in xrange(1, mq):
                steps.append((i, j))
        for p, q in steps:
            arr = np.zeros((N, M))
            for i in xrange(self.N):
                for j in xrange(self.M):
                    arr[i, j] = ksp[p](ksp, i) * ksq[q](ksq, j) / (ros[p]*ros[q])
            self.Tk.append(arr)
        '''for x in self.Tk:
            print x
        for x in ros:
            print x
        '''
    def __call__(self, arr):
        def calcOne(x): return sum(sum(x * arr))
        return map(calcOne, self.Tk)

    def getRos(self, max_p):
        res = []
        N = float(self.N)
        cur = N
        for i in xrange(max_p + 1):
            cur *= 1 - (i**2)/(N**2)
            #print cur, i, N
            res.append(cur/(2*i - 1))
        return res

