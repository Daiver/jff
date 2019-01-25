import numpy as np
import sys

def readMTXMatrix(fname):
    f = open(fname)
    header = f.readline()
    params = f.readline()
    nRows, nCols, nItems = map(int, params.split(' '))
    res = np.zeros((nRows, nCols), dtype=np.float32)
    for s in f:
        tokens = s.split(' ')
        row, col = map(int, tokens[0:2])
        val = float(tokens[2])
        res[row - 1, col - 1] = val
    f.close()
    return res

def readVec(fname):
    f = open(fname)
    nItems = int(f.readline())
    res = np.zeros(nItems, dtype=np.float32)
    for i in xrange(nItems):
        res[i] = float(f.readline())
    f.close()
    return res

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

if __name__ == '__main__':
    mat = readMTXMatrix(sys.argv[1])
#    vec = readVec(sys.argv[2])
    #ans = readVec(sys.argv[3])
    #vars = np.linalg.solve(mat, vec)
    #for i in xrange(ans.shape[0]):
        #diff = np.abs(vars[i] - ans[i])
        #if diff > 0.001:
            #print i, diff, vars[i], ans[i]
    #print sum(mat.dot(ans) - vec)
    #print sum(mat.dot(vars) - vec)
    print (mat.transpose() == mat).all()
    print is_pos_def(mat)
    print min(np.linalg.eigvals(mat))
    print np.linalg.det(mat)
