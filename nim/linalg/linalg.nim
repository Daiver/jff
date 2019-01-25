
import matrix
include lapacke_backend
from matrix import Matrix, fromValues

proc lstsquares*(aMat: Matrix[float64], bVec: Matrix[float64]): Matrix[float64] = 
    let a = aMat.t * aMat
    let b = aMat.t * bVec
    return solveSystemOfLinearEquations(a, b)


proc fuzzyComp*[T](a: T, b: T): bool = return (a - b).abs() < 0.00001
proc fuzzyComp*(a: int, b: int): bool = return a == b
proc fuzzyComp*[T](a: matrix.Matrix[T], b: matrix.Matrix[T]): bool = 
    let rows = a.rows
    let cols = a.cols
    if rows != b.rows or cols != b.cols:
        return false
    for row in 0..rows - 1:
        for col in 0..cols - 1:
            if not fuzzyComp(a[row, col], b[row, col]):
                return false
    return true

