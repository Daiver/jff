import Data.Maybe
import qualified Data.Map as M
import Estimator
import Tests
import Definitions
import BuildInFunctions

main = do
    
    print $ estimate example (M.fromList [("x", 10)])
    print $ simplify $ partialDerr "x" (sq (Variable "x"))
    print $ simplify $ partialDerr "x" example3
    print $ estimate (partialDerr "x" example) (M.fromList [("x", 10)])
    print $ estimate (partialDerr "x" example3) (M.fromList [("x", 10)])
    print $ (subOp [Constant 1, Operation add [Variable "y", Constant 5]])
    print $  simplify $ partialDerr "x" (subOp [Constant 1, Operation add [Variable "y", Constant 5]])
    --print $ estimate (Operation (Function "cos" (cos . head)) [Variable "x"]) M.empty
    print $ partialDerr "x" (Operation (UnaryOperator "cos" (cos )) [Variable "x"])
    print$ simplify $ partialDerr "x" example6
