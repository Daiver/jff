import sequtils
import future

type
    Matrix *[Scalar] = ref object
        values : seq[Scalar]
        rows : int
        cols : int

proc rows * [T](mat: Matrix[T]): int = return mat.rows
proc cols * [T](mat: Matrix[T]): int = return mat.cols
proc values * [T](mat: Matrix[T]): seq[T] = return mat.values

proc `[]` * [T] (mat: Matrix[T], row: int, col: int): T {.inline.} =
    return mat.values[row * mat.cols + col]

proc `[]` * [T] (mat: Matrix[T], row: int): T {.inline.} =
    assert mat.cols == 1
    return mat.values[row]

proc `[]=` * [T] (mat: var Matrix[T], row: int, col: int, value: T) {.inline.} =
    mat.values[row * mat.cols + col] = value

proc `[]=` * [T] (mat: var Matrix[T], row: int, value: T) {.inline.} =
    assert mat.cols == 1
    mat.values[row] = value

proc zeros*[T](rows, cols: int): Matrix[T] =
    var values : seq[T] = @[]
    values.setLen(rows * cols)
    return Matrix[T](values: values, rows: rows, cols: cols)

proc zeros*[T](rows: int): Matrix[T] =
    var values : seq[T] = @[]
    values.setLen(rows)
    return Matrix[T](values: values, rows: rows, cols: 1)

# should be replaced by macro

proc `+` * [T](mat1, mat2: Matrix[T]): Matrix[T] {.inline.} =
    var valR = mat1.values
    for i in 0 .. mat1.values.len - 1:
        valR[i] = mat1.values[i] + mat2.values[i]
    return Matrix[T](values: valR, rows: mat1.rows, cols: mat1.cols)

proc `+` * [T](mat1: Matrix[T], a: T): Matrix[T] {.inline.} =
    var valR = mat1.values
    for i in 0 .. mat1.values.len - 1:
        valR[i] = mat1.values[i] + a
    return Matrix[T](values: valR, rows: mat1.rows, cols: mat1.cols)

proc `+` * [T](a: T, mat1: Matrix[T]): Matrix[T] {.inline.} =
    var valR = mat1.values
    for i in 0 .. mat1.values.len - 1:
        valR[i] = mat1.values[i] + a
    return Matrix[T](values: valR, rows: mat1.rows, cols: mat1.cols)

proc `-` * [T](mat1, mat2: Matrix[T]): Matrix[T] {.inline.} =
    var valR = mat1.values
    for i in 0 .. mat1.values.len - 1:
        valR[i] = mat1.values[i] - mat2.values[i]
    return Matrix[T](values: valR, rows: mat1.rows, cols: mat1.cols)

proc `-` * [T](mat1: Matrix[T], a: T): Matrix[T] {.inline.} =
    var valR = mat1.values
    for i in 0 .. mat1.values.len - 1:
        valR[i] = mat1.values[i] - a
    return Matrix[T](values: valR, rows: mat1.rows, cols: mat1.cols)

proc `-` * [T](a: T, mat1: Matrix[T]): Matrix[T] {.inline.} =
    var valR = mat1.values
    for i in 0 .. mat1.values.len - 1:
        valR[i] = a - mat1.values[i] 
    return Matrix[T](values: valR, rows: mat1.rows, cols: mat1.cols)

proc `.*` * [T](mat1, mat2: Matrix[T]): Matrix[T] {.inline.} =
    var valR = mat1.values
    for i in 0 .. mat1.values.len - 1:
        valR[i] = mat1.values[i] * mat2.values[i]
    return Matrix[T](values: valR, rows: mat1.rows, cols: mat1.cols)

proc `.*` * [T](mat1: Matrix[T], a: T): Matrix[T] {.inline.} =
    var valR = mat1.values
    for i in 0 .. mat1.values.len - 1:
        valR[i] = mat1.values[i] * a
    return Matrix[T](values: valR, rows: mat1.rows, cols: mat1.cols)

proc `.*` * [T](a: T, mat1: Matrix[T]): Matrix[T] {.inline.} =
    var valR = mat1.values
    for i in 0 .. mat1.values.len - 1:
        valR[i] = mat1.values[i] * a
    return Matrix[T](values: valR, rows: mat1.rows, cols: mat1.cols)

proc `./` * [T](mat1, mat2: Matrix[T]): Matrix[T] {.inline.} =
    var valR = mat1.values
    for i in 0 .. mat1.values.len - 1:
        valR[i] = mat1.values[i] / mat2.values[i]
    return Matrix[T](values: valR, rows: mat1.rows, cols: mat1.cols)

proc `*` * [T](mat1, mat2: Matrix[T]): Matrix[T] {.inline.} =
    assert mat1.cols == mat2.rows
    let rows = mat1.rows
    let cols = mat2.cols

    let inner = mat1.cols

    var res = zeros[T](rows, cols)
    for row in 0 .. rows-1:
        for col in 0 .. cols-1:
            for k in 0 .. inner-1:
                res[row, col] = res[row, col] + mat1[row, k] * mat2[k, col]
    return res

proc sum * [T](mat: Matrix[T]): T = 
    return mat.values.foldl(a + b)

#non effective
proc reshape * [T](mat: Matrix[T], rows: int, cols: int): Matrix[T] {.inline.} = 
    return Matrix[T](values: mat.values, rows: rows, cols: cols)

#non effective
proc transposed * [T](mat: Matrix[T]): Matrix[T] {.inline.} = 
    var res = zeros[T](mat.cols, mat.rows)
    for row in 0..mat.rows - 1:
        for col in 0..mat.cols - 1:
            res[col, row] = mat[row, col]
    return res

proc t * [Scalar](mat: Matrix[Scalar]): Matrix[Scalar] {.inline.} = 
    return mat.transposed()

proc fromValues * [T](values: seq[T]): Matrix[T] =
    return Matrix[T](values: values, rows: values.len, cols: 1)

proc fromValues * [T, Rows](values: array[Rows, T]): Matrix[T] =
    return Matrix[T](values: @values, rows: values.len, cols: 1)

proc fromValues * [T](values: seq[seq[T]]): Matrix[T] =
    return Matrix[T](values: sequtils.foldl(values, a & b), 
                     rows: values.len, 
                     cols: values[0].len)

#proc fromValues * [T, Rows, Cols](values: array[Rows, array[Cols, T]]): Matrix[T] =
    #return Matrix[T](values: values.foldl((a, b) => (@a) & (@b)), 
                     #rows: values.len, 
                     #cols: values[0].len)

proc `$` * [T](mat: Matrix[T]): string =
    var res = "["
    for row in 0 .. mat.rows - 1:
        if row != 0:
            res.add(" ")
        res.add("[ ")
        for col in 0 .. mat.cols - 1:
            res.add(mat[row, col].`$`)
            res.add(" ")
        res.add("]")
        if row != mat.rows - 1:
            res.add("\n")
    res.add("]")
    return res
