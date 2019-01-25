import sequtils
import algorithm
import future

import ../linalg/matrix.nim


proc argsort *[T](values: Matrix[T]): seq[int] = 
    assert values.cols() == 1
    var indices = toSeq 0..values.rows() - 1
    indices.sort do (x, y: int) -> int:
        return cmp(values[x], values[y])
    return indices
