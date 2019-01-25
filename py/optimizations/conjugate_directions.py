import numpy as np

def gradientDescentForQuad1(nIter, A, b, x0):
    x = x0
    for i in xrange(nIter):
        r = b - A.dot(x)
        x += 0.1 * r
        print sum((b - A.dot(x)) ** 2), r, x

def gradientDescentForQuad2(nIter, A, b, x0):
    x = np.array(x0, dtype=np.float32)
    for i in xrange(nIter):
        r = b - A.dot(x)
        
        a = r.dot(r) / (r.T.dot(A).dot(r))
        x += a * r
        print sum((b - A.dot(x)) ** 2), r, x


if __name__ == '__main__':
    A = np.array([
        [3, 2],
        [2, 6]
        ])
    B = np.array([2, -8])
    print A.dot([2, -2]) - B
    #gradientDescentForQuad1(20, A, B, [-2, -2])
    gradientDescentForQuad2(20, A, B, [-2, -2])
