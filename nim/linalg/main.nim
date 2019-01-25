import linalg
import matrix

var a = matrix.fromValues([1, 2, 3])
var b = a.t
b[0, 0] = 10

echo($a)
echo($b)
