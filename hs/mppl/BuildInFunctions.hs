module BuildInFunctions where

import Definitions

add, mul, pow :: Function Float

add = BinaryOperator "+" (+) 
mul = BinaryOperator "*" (*) 
pow = BinaryOperator "^" (**)

buildInFunctions = [add, mul, pow]

negOp :: Operation Float -> Operation Float
negOp x = Operation mul [Constant (negate 1.0), x]
subOp (a:b:_) = Operation add [a, negOp b]

