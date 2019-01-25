import matrix
from matrix import Matrix, fromValues
import linalg

let a = fromValues([
        1.0, 1, 
        3, 1,
        4, 1]).reshape(3, 2)

let b = fromValues([
        1.0, 3, 4
    ])

let solution = linalg.lstsquares(a, b)

echo($solution)
