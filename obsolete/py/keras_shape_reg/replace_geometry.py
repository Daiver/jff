import numpy as np
import sys

def readObj2Lines(fname):
    res = []
    with open(fname) as f:
        for s in f:
            res.append(s)
    return res

def replaceGeometry(linesOfOriginalObj, newVertices):
    counter = 0
    res = []

    for line in linesOfOriginalObj:
        if len(line) == 0 or line[0:2] != 'v ':
            res.append(line)
            continue
        vertex = newVertices[counter]
        res.append('v %s %s %s\n' % (str(vertex[0]), str(vertex[1]), str(vertex[2])))
        counter += 1
    assert counter == len(newVertices)

    return res

def replaceGeometryAndWrite(linesOfOriginalObj, newVertices, fname):
    newLines = replaceGeometry(linesOfOriginalObj, newVertices)
    with open(fname, 'w') as f:
        for l in newLines:
            f.write(l)

if __name__ == '__main__':
    originalObj = readObj2Lines('./cube.obj')
    v = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 25]
        ]
    replaceGeometryAndWrite(originalObj, v, 'res.obj')
