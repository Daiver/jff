import numpy as np
import cv2 

def drawPoints(positions, adj, width, height):
    img = np.ones((width, height, 3), dtype=np.uint8)
    img *= 255
    for i, a in enumerate(adj):
        p1 = positions[i]
        for j in a:
            p2 = positions[j]
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0))

    for p in positions:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 0, 0))

    return img

def readMSMMFromFile(fname, scale=1.0):
    with open(fname, 'rt') as f:
        nVerts = int(f.readline())
        positions = np.zeros((nVerts, 2), dtype=np.float32)
        for i in xrange(nVerts):
            s = map(float, f.readline().split())
            positions[i, 0] = s[0]*scale
            positions[i, 1] = s[1]*scale
        adj = []
        for i in xrange(nVerts):
            s = map(int,f.readline().split())
            adj.append(s)

    return adj, positions

def printVec(v):
    return ', '.join('[' + ('%.5f ' % x[0]) + ('%.5f' % x[1]) + ']' for x in v)

def findCommonVertices(adj, startInd, endInd):
    return filter(lambda x: x != endInd and x in adj[endInd], adj[startInd])

def cellIndices(adj):
    res = []
    for i, a in enumerate(adj):
        for j in a:
            res.append([i, j] + findCommonVertices(adj, i, j))
    return res

def edgeLengths(pos, adj):
    res = []
    for i, a in enumerate(adj):
        for j in a:
            res.append(pos[j] - pos[i])
    return np.array(res)

def gMatrix(pos, cellInd):
    res = np.zeros((len(cellInd*2), 2))
    for i, ind in enumerate(cellInd):
        res[2*i, 0] = pos[ind, 0]
        res[2*i, 1] = pos[ind, 1]
        res[2*i+1, 0] = pos[ind, 1]
        res[2*i+1, 1] = -pos[ind, 0]

    return res

def gMatrices(pos, cellInds):
    return [gMatrix(pos, x) for x in cellInds]

def hMatrix(edgeLen, lengthOfCell, g):
    eye = np.zeros((2, 2*lengthOfCell))
    eye[0, 0] = -1
    eye[1, 1] = -1
    eye[0, 2] = 1
    eye[1, 3] = 1

    e = np.array([
        [edgeLen[0],  edgeLen[1]],
        [edgeLen[1], -edgeLen[0]]
        ])

    g2 = np.linalg.pinv(np.dot(g.transpose(), g))
    g3 = np.dot(e, g2)
    g4 = np.dot(g3, g.transpose())
    return eye - g4

def hMatrices(eLengths, cellLengths, gMats):
    return [hMatrix(e, c, g) for e, c, g in zip(eLengths, cellLengths, gMats)]

def composeA1Matrix(hMats, cells, nVerts, weight, constraintsInds):
    rows = 2*len(hMats) + 2*len(constraintsInds)
    cols = 2*nVerts
    A = np.zeros((rows, cols))
    for i, (h, c) in enumerate(zip(hMats, cells)):
        for j, cInd in enumerate(c):
            A[2*i,     2*cInd    ] = h[0, 2*j]
            A[2*i,     2*cInd + 1] = h[0, 2*j + 1]
            A[2*i + 1, 2*cInd    ] = h[1, 2*j]
            A[2*i + 1, 2*cInd + 1] = h[1, 2*j + 1]
    startInd = 2*len(hMats)
    for i, ind in enumerate(constraintsInds):
        A[startInd + 2*i,     2*ind    ] = weight
        A[startInd + 2*i + 1, 2*ind + 1] = weight
    return A

def composeB1Matrix(nCells, weight, constraints):
    B = np.zeros((2*nCells + 2*len(constraints)))
    startInd = 2*nCells
    for i, val in enumerate(constraints):
        B[startInd + 2*i    ] = weight*val[0]
        B[startInd + 2*i + 1] = weight*val[1]
    return B

def computeARAPImageWarpStage1(
        originalPositions, 
        adjacentList, 
        controlPointsIndices, 
        controlPointDesirePositions,
        weight):
    edgeLens = edgeLengths(originalPositions, adjacentList) 
    cells    = cellIndices(adjacentList)
    gs       = gMatrices(originalPositions, cells)
    hs       = hMatrices(edgeLens, map(len, cells), gs)
    A        = composeA1Matrix(
                    hs, 
                    cells, 
                    len(originalPositions), 
                    weight, 
                    controlPointsIndices)
    B        = composeB1Matrix(len(cells), weight, controlPointDesirePositions)
    print 'A shape', A.shape
    np.set_printoptions(threshold=np.nan,precision=5)

    #print B
    return np.linalg.lstsq(A, B)[0].reshape((-1, 2)), gs, cells

def normalizedTransformationFromPositions(newPos, g, cell):
    cords = []
    for ind in cell:
        cords.append(newPos[ind][0])
        cords.append(newPos[ind][1])
    T = np.dot(np.dot(np.linalg.pinv(np.dot(g.transpose(), g)), g.transpose()), cords)
    ck, sk = T.tolist()
    norm = 1.0/(np.sqrt((ck**2 + sk**2)))
    #print 'ck', ck, 'sk', sk, 'T', T, 'TPrime', TPrime
    return np.array([
        [ck*norm, sk*norm],
        [-sk*norm, ck*norm],
        ])

def composeA2Matrix(cells, constraintsInds, nVerts, weight):
    A = np.zeros((len(cells) + len(constraintsInds), nVerts))
    for i, c in enumerate(cells):
        A[i, c[0]] = -1
        A[i, c[1]] =  1
    startInd = len(cells)
    for i, ind in enumerate(constraintsInds):
        A[startInd + i, ind] = weight
    return A

def composeB2Matrix(edgeLens1D, controlPointDesirePositions1D, weight):
    B = np.zeros((len(edgeLens1D) + len(controlPointDesirePositions1D)))
    for i, e in enumerate(edgeLens1D):
        B[i] = e
    for i, c in enumerate(controlPointDesirePositions1D):
        B[len(edgeLens1D) + i] = c*weight
    return B

def computeARAPImageWarpStage2(
        oldPos, newPos, 
        adj, gs, cells, 
        controlPointsIndices, controlPointDesirePositions,
        weight):
    transforms = [normalizedTransformationFromPositions(newPos, g, c) 
                  for g, c in zip(gs, cells)]
    edgeLens = edgeLengths(oldPos, adj) 
    transformedOldEdgeLens = np.array([np.dot(t, e) for t, e in zip(transforms, edgeLens)])
    controlPointDesirePositions = np.array(controlPointDesirePositions)
    A    = composeA2Matrix(cells, controlPointsIndices, len(oldPos), weight)
    BX   = composeB2Matrix(transformedOldEdgeLens[:, 0], controlPointDesirePositions[:, 0], weight)
    BY   = composeB2Matrix(transformedOldEdgeLens[:, 1], controlPointDesirePositions[:, 1], weight)
    #BX   = composeB2Matrix(edgeLens[:, 0], controlPointDesirePositions[:, 0], weight)
    #BY   = composeB2Matrix(edgeLens[:, 1], controlPointDesirePositions[:, 1], weight)
    resX = np.linalg.lstsq(A, BX)[0]
    resY = np.linalg.lstsq(A, BY)[0]

    '''print 'trans'
    for t in A:
        print t
    print '''
    #print A
    #print len(cells)
    res = cv2.merge([resX.reshape(-1), resY.reshape(-1)])
    print res.shape, resX.shape
    return res.reshape((-1, 2))

def nearestPoint(pos, x, y):
    cur = np.array([x, y])
    bestDst = 1e6
    bestInd = -1
    for i, p in enumerate(pos):
        dst = (p[0] - cur[0])**2 + (p[1] - cur[1])**2
        if dst < bestDst:
            bestDst = dst
            bestInd = i
    return bestInd, bestDst

def mouseCallback(event, x, y, flags, data):
    pos, adj, controlPointsIndices, controlPointDesirePositions, newPos, selectedInd = data
    if event == cv2.EVENT_LBUTTONDOWN:
        nearestInd, nearestDst = nearestPoint(newPos, x, y)
        print flags, nearestInd
        if flags & cv2.EVENT_FLAG_SHIFTKEY :
            toFind = controlPointDesirePositions
            nearestInd,nearestDst = nearestPoint(toFind, x, y)
            print nearestInd, 'selected'
            selectedInd = nearestInd

            #print newPos
        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            if nearestInd not in controlPointsIndices and nearestDst < 50.1:
                controlPointsIndices.append(nearestInd)
                print nearestInd, 'added'
                selectedInd = len(controlPointDesirePositions)
                controlPointDesirePositions.append(np.array([x, y]))
        else:
            controlPointDesirePositions[(selectedInd)] = [x, y]
            
            newPos, gs, cells = computeARAPImageWarpStage1(pos, adj, controlPointsIndices, controlPointDesirePositions, 100000)
            newPos = computeARAPImageWarpStage2(pos, newPos, adj, gs, cells, controlPointsIndices, controlPointDesirePositions, 100000)


            
        img = drawPoints(newPos, adj, 1200, 1200)
        cv2.imshow('', img)
        cv2.setMouseCallback('', mouseCallback, 
                [pos, adj, 
                controlPointsIndices, controlPointDesirePositions, newPos,
                selectedInd])

if __name__ == '__main__':
    #adj, pos = readMSMMFromFile('./testSimplePoints1.txt')
    #adj, pos = readMSMMFromFile('./simplepoints2.txt')
    adj, pos = readMSMMFromFile('./simplepoints3.txt', 1.5)
    
    img = drawPoints(pos, adj, 1200, 1200)
    cv2.imshow('before', img)

    controlPointsIndices        = [
            #8, 
            4, 
            0, 1, 2, 3
            ]
    controlPointDesirePositions = [
            #pos[8], 
            pos[4], 
            pos[0], pos[1], pos[2], pos[3]
            ]

    newPos, gs, cells = computeARAPImageWarpStage1(pos, adj, controlPointsIndices, controlPointDesirePositions, 100000)
    #print newPos

    newPos = computeARAPImageWarpStage2(pos, newPos, adj, gs, cells, controlPointsIndices, controlPointDesirePositions, 100000)

    pos = newPos
    
    img = drawPoints(pos, adj, 1200, 1200)
    cv2.imshow('', img)
    cv2.setMouseCallback('', mouseCallback, 
            [pos, adj, 
            controlPointsIndices, controlPointDesirePositions, pos, 4])

    key = cv2.waitKey()
    while key % 0x100 != 27:
        key = cv2.waitKey()

