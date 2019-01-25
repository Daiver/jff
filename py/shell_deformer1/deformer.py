import numpy as np

def composeL1Matrix(verticesAdjacency):
    nVertices = len(verticesAdjacency)
    matrix = np.zeros((nVertices, nVertices))
    for vInd in xrange(nVertices):
        adjs = verticesAdjacency[vInd]
        matrix[vInd, vInd] -= len(adjs)
        for vInd2 in adjs:
            matrix[vInd, vInd2] += 1
    return matrix

def composeL2Matrix(verticesAdjacency, l1Mat):
    nVertices = len(verticesAdjacency)
    matrix = np.zeros((nVertices, nVertices))
    for vInd in xrange(nVertices):
        adjs = verticesAdjacency[vInd]
        for vInd2 in adjs:
            for vInd2P in adjs:
                matrix[vInd, vInd ] += 1
                matrix[vInd, vInd2] -= 1
            adjs2 = verticesAdjacency[vInd2]
            for vInd3 in adjs2:
                matrix[vInd, vInd2] -= 1
                matrix[vInd, vInd3] += 1
                #print vInd, vInd2, vInd3, matrix[vInd, :]

       
    return matrix

def composeRHS(aMat, hardConstraintsIndices, hardConstraintsDisplacements1D):
    nVertices = aMat.shape[0]
    nFixedVertices = len(hardConstraintsIndices)
    nFinalVertices = nVertices - nFixedVertices
    B = np.zeros(nFinalVertices)
    assert(nFixedVertices == len(hardConstraintsDisplacements1D))
    
    for i, index in enumerate(hardConstraintsIndices):
        row = aMat[:, index]
        offset = 0
        for col in xrange(row.shape[0]):
            if col in hardConstraintsIndices:
                continue
            B[offset] -= hardConstraintsDisplacements1D[i] * row[col]
            offset += 1
        #B += hardConstraintsDisplacements1D[i] * aMat[:, index]

    return B

def composeAMatFinal(aMat, hardConstraintsIndices):
    nVertices = aMat.shape[0]
    nFixedVertices = len(hardConstraintsIndices)
    nFinalVertices = nVertices - nFixedVertices
    A = np.zeros((nFinalVertices, nFinalVertices))

    rowOffset = 0
    for row in xrange(nVertices):
        if row in hardConstraintsIndices:
            continue
        colOffset = 0
        for col in xrange(nVertices):
            if col in hardConstraintsIndices:
                continue
            A[rowOffset, colOffset] = aMat[row, col]
            colOffset += 1
        rowOffset += 1

    return A

def composeFinalDisplacements1D(hardConstraintsIndices, hardConstraintsDisplacements1D, displacements):
    nVertices = len(hardConstraintsIndices) + len(displacements)
    res = np.zeros(nVertices)
    offset = 0
    for i in xrange(nVertices):
        if i in hardConstraintsIndices:
            res[i] = hardConstraintsDisplacements1D[hardConstraintsIndices.index(i)]
            continue
        res[i] = displacements[offset]
        offset += 1
    return res
