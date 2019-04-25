import itertools
import openturns
import numpy

class KrawtchoukDesc:
    def __init__(self, p1, p2, size):
        self.size = size
        self.KF1 = openturns.KrawtchoukFactory(size[0] - 1, p1)
        self.KF2 = openturns.KrawtchoukFactory(size[1] - 1, p2)
        steps = []
        for i in xrange(5):
            for j in xrange(5):
                steps.append((i, j))
        self.Qf = [(self.KF1.build(n), self.KF2.build(m)) for n, m in steps]
        self.q = []
        N, M = self.size
        for qf in self.Qf:
            K1, K2 = qf
            arr = numpy.zeros(self.size)
            for x in xrange(N):
                for y in xrange(M):
                    arr[x, y] = K1(x) * K2(y)
            self.q.append(arr)

    def __call__(self, arr):
        if arr.shape != self.size: raise Exception('BAD SIZE %s %s' % (self.size, arr.shape))
        N, M = arr.shape
        def calcOne(q):
            res = sum(sum(q * arr))
            return res
        return map(calcOne, self.q)

class OldKrawtchoukDesc:
    def __init__(self, p1, p2, size):
        self.size = size
        self.KF1 = openturns.KrawtchoukFactory(size[0] - 1, p1)
        self.KF2 = openturns.KrawtchoukFactory(size[1] - 1, p2)
        steps = []
        for i in xrange(5):
            for j in xrange(5):
                steps.append((i, j))
        self.Q = [(self.KF1.build(n), self.KF2.build(m)) for n, m in steps]

    def __call__(self, arr):
        if arr.shape != self.size: raise Exception('BAD SIZE %s %s' % (self.size, arr.shape))
        N, M = arr.shape

        def calcOne(q):
            K1, K2 = q
            sum = 0
            for x in xrange(N):
                for y in xrange(M):
                    sum += K1(x) * K2(y) * arr[x, y]
            return sum
        return numpy.array(map(calcOne, self.Q))


