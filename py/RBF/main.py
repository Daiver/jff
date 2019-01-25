import numpy as np

def RBFCoeffsFromPoints(points):
    X = [p[0] for p in points]
    Y = [p[1] for p in points]
    r = np.zeros((len(points)), np.float32)
    for i, p1 in enumerate(X):
        minVal = 1e6
        for j, p2 in enumerate(X):
            if i == j: continue
            if np.abs(p1 - p2) < minVal:
                minVal = np.abs(p1 - p2)
        r[i] = minVal
    
    A = np.zeros((len(points), len(points)), dtype=np.float32)
    B = np.zeros((len(points)), dtype=np.float32)
    for j in xrange(len(points)):
        for i in xrange(len(points)):
            A[j, i] = (X[i] - X[j])**2 + r[i]**2
    for j in xrange(len(points)):
        B[j] = Y[j]

    print A
    print B
    print r
    return X, r, np.linalg.solve(A, B)

def RBFCoeffsLSM(points):
    X = [p[0] for p in points]
    Y = [p[1] for p in points]
    r = np.zeros((len(points)), np.float32)
    for i, p1 in enumerate(X):
        minVal = 1e6
        for j, p2 in enumerate(X):
            if i == j: continue
            if np.abs(p1 - p2) < minVal:
                minVal = np.abs(p1 - p2)
        r[i] = minVal
    
    A = np.zeros((len(points), len(points) + 2), dtype=np.float32)
    B = np.zeros((len(points)), dtype=np.float32)
    for j in xrange(len(points)):
        for i in xrange(len(points)):
            A[j, i] = (X[i] - X[j])**2 + r[i]**2
            A[j, len(points) - 2] = X[i]
            A[j, len(points) - 1] = 1
    for j in xrange(len(points)):
        B[j] = Y[j]
    return X, r, np.linalg.lstsq(A,B)

def RBF(X, r, alpha):
    return lambda x: sum(
            alpha[i]*((x - X[i])**2 + r[i]**2) 
            for i in xrange(alpha.shape[0] - 2)) + x*alpha[-2] + alpha[-1]

if __name__ == '__main__':
    points = [[1, 2], [2, 3], [0, 4], [2.5, 5], [3, 7]]
    X, r, ans = RBFCoeffsLSM(points)
    print 'ans'
    print ans
    rbf = RBF(X, r, ans[0])
    for x, y in points:
        print rbf(x), y
