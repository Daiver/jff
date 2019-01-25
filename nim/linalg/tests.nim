import unittest
import matrix
import lapacke_backend

proc fuzzyComp[T](a: T, b: T): bool = return (a - b).abs() < 0.00001
proc fuzzyComp(a: int, b: int): bool = return a == b
proc fuzzyComp[T](a: matrix.Matrix[T], b: matrix.Matrix[T]): bool = 
    let rows = a.rows
    let cols = a.cols
    if rows != b.rows or cols != b.cols:
        return false
    for row in 0..rows - 1:
        for col in 0..cols - 1:
            if not fuzzyComp(a[row, col], b[row, col]):
                return false
    return true

suite "Matrix tests":

    test "sum test 01":
        let mat = matrix.fromValues([1.0, 2, 3.0])
        let sm = mat.sum()
        check(fuzzyComp(sm, 6.0))

    test "add test 01":
        let mat1 = matrix.fromValues([1.0, 2, 3, 4, 5, 6]).reshape(2, 3)
        let mat2 = matrix.fromValues([5.0, 6, 7, 8, 9, 10]).reshape(2, 3)
        let mat3 = mat1 + mat2
        let mat4 = matrix.fromValues([6.0, 8, 10, 12, 14, 16]).reshape(2, 3)
        check(fuzzyComp(mat3, mat4))
 
    test "add test 02":
        let mat1 = matrix.fromValues([1.0, 2, 3, 4, 5, 6]).reshape(2, 3)
        let mat3 = mat1 + 10
        let mat4 = matrix.fromValues([11.0, 12, 13, 14, 15, 16]).reshape(2, 3)
        check(fuzzyComp(mat3, mat4))

    test "add test 03":
        let mat1 = matrix.fromValues([1.0, 2, 3, 4, 5, 6]).reshape(2, 3)
        let mat3 = 3.0 + mat1 + 7
        let mat4 = matrix.fromValues([11.0, 12, 13, 14, 15, 16]).reshape(2, 3)
        check(fuzzyComp(mat3, mat4))

    test "sub test 01":
        let mat1 = matrix.fromValues([1.0, 2, 3, 4, 5, 6]).reshape(2, 3)
        let mat2 = matrix.fromValues([5.0, 6, 7, 8, 9, 10]).reshape(2, 3)
        let mat3 = mat1 - mat2
        let mat4 = matrix.fromValues([-4.0, -4, -4, -4, -4, -4]).reshape(2, 3)
        check(fuzzyComp(mat3, mat4))

    test "mul test 01":
        let mat1 = matrix.fromValues([1.0, 2, 3, 4, 5, 6]).reshape(3, 2)
        let mat2 = matrix.fromValues([5.0, 6, 7, 8]).reshape(2, 2)
        let mat3 = mat1 * mat2
        let mat4 = matrix.fromValues([19.0, 22, 43, 50, 67, 78]).reshape(3, 2)
        check(fuzzyComp(mat3, mat4))
  
    test "transpose test 01":
        let mat1 = matrix.fromValues([1, 2, 3, 4, 5, 6]).reshape(2, 3)
        let mat2 = matrix.fromValues([1, 4, 2, 5, 3, 6]).reshape(3, 2)
        check(fuzzyComp(mat1.t(), mat2))

    test "solve test 01":
        let matA = matrix.fromValues([
               6.80, -6.05, -0.45,  8.32, -9.67,
               -2.11, -3.30,  2.58,  2.71, -5.14,
                5.66, 5.36, -2.70,  4.35, -7.26,
                5.97, -4.44,  0.27, -7.17, 6.08,
                8.23, 1.08,  9.04,  2.14, -6.87
            ]).reshape(5, 5)
        let matB = matrix.fromValues([
                4.02, 
                6.19,
               -8.22,
               -7.57,
               -3.03 
            ])
        let res = lapacke_backend.solveSystemOfLinearEquations(matA, matB)
        let ans = matrix.fromValues([
             -0.800714026,
             -0.695243384,
              0.593914995,
              1.321725609,
              0.565756197
        ])
        check(fuzzyComp(res, ans))
