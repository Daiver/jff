import cinter
import matrix

proc solveSystemOfLinearEquations*(
    aMat: ptr float64, 
    bVec: ptr float64, 
    n: cint, 
    resVec: ptr float64) {.importc.}

proc solveSystemOfLinearEquations*(
        matA: matrix.Matrix[float64],
        matB: matrix.Matrix[float64]): Matrix[float64] = 
    assert matA.rows() == matA.cols()
    assert matA.rows() == matB.rows()
    #var resMat: matrix.Matrix[float64] = matrix.zeros[float64](matA.rows(), 1)
    var res: seq[float64] = @[]
    res.setLen(matA.rows())
    let valuesA = matA.values() 
    let valuesB = matB.values() 
    solveSystemOfLinearEquations(
            cinter.seqPointer(valuesA),
            cinter.seqPointer(valuesB),
            cint(matA.rows()),
            cinter.seqPointer(res))

    return matrix.fromValues[float64](res)
