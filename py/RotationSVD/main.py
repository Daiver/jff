import numpy as np

if __name__ == '__main__':
    pointsSource = np.array([
        [1, 2, 3],
        #[1, 1, 1],
        ], dtype=np.float32)
    pointsTarget = np.array([
        [10, -20, 30],
        #[1, 1, -1],
        ], dtype=np.float32)
    A = np.dot(pointsSource.transpose(), pointsTarget)
    print 'A'
    print A
    u, s, v = np.linalg.svd(A)
    print 'res'
    print np.dot(np.dot(u, np.diag(s)), v)
    print ''
    print u
    print s
    print v

    R = np.dot(v.transpose(), u.transpose())
    print R
    print 'det(R)'
    print np.linalg.det(R)
    print np.dot(R, pointsSource.transpose())
