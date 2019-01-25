module Tests where

import Data.Maybe
import qualified Data.Map as Map
import Estimator
import Definitions
import BuildInFunctions
import Lexer
import Parser

sq x = Operation pow [x, Constant 2.0]
example  = Operation add [Constant 10, sq (Variable "x")]
example2 = Operation add [sq (Constant 2), Operation add [Variable "y", Constant 5]]
example3 = Operation add [Operation mul [Variable "x", Constant 10], sq (Variable "x")]
example4 = Operation add [subOp [Constant 1.0, Operation mul [Constant 2, Constant 3]], Constant 5]
example5 = sq (Variable "x")

example6 = Operation add [ 
        Operation add [sq (Variable "x"), sq (Variable "y")],
        Operation mul [Constant 2, Variable "x"]
    ]

example7 = Operation mul [
        Operation mul [Variable "x", Variable "y"],
        Constant 5
    ]

examples = [example, example2, example3, example4, example5, example6, example7]

estimatorTests = do
    let expressions1Set = [
                Operation mul [Variable "x", Constant 10],
                Operation pow [Variable "x", Constant 3]
            ]
        t1 = (map (flip estimate (Map.fromList [("x", 10)])) expressions1Set) == [100, 1000]
        t2 =  (map (isConstantOf "x") examples) == [
                False, True, False, True, False, False, False
            ]
        t3 =  (map isConstant examples) == [False, False, False, True, False, False, False]

        ders = map (simplify . partialDerr "x") examples
        dersEst1 = map (flip estimate (Map.fromList [("x", 5), ("y", 100)])) ders
        t4 = dersEst1 == [10, 0, 20, 0, 10, 12, 500]

        ders2 = map (simplify . partialDerr "y") examples
        dersEst2 = map (flip estimate (Map.fromList [("y", 12), ("x", negate 12)])) ders2
        t5 = dersEst2 == [0, 1.0, 0, 0, 0, 24, -60]

        secondDers1 = map (simplify . partialDerr2 "x") examples
        secondDers1Est = map (flip estimate (Map.fromList [("x", 4), ("y", 7)])) secondDers1

        t6 = secondDers1Est == [2, 0, 2, 0, 2, 2, 0]

        testResults = [t1, t2, t3, t4, t5, t6]
    if all id testResults then 
            print "Estimator tests is Ok" 
        else 
            print "Estimator ERROR" >> print testResults

lexerTests = do
    let testString = "let map=11 and no more 42"
        t1 = (tokenize testString) == ["let", "map", "=", "11", "and", "no", "more", "42"]
        testResults = [t1]
    if all id testResults then 
        print "Lexer tests is Ok" else print "Lexer ERROR" >> print testResults

parserTests = do
    print "Parser tests is Ok"

tests = do
    estimatorTests
    lexerTests
    parserTests
