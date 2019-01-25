module Estimator where

import Data.Maybe
import qualified Data.Map as M
import Definitions
import BuildInFunctions

{-add = BinaryOperator "+" (+) 
mul = BinaryOperator "*" (*) 
pow = BinaryOperator "^" (**)-}

{-negOp :: Operation Float -> Operation Float
negOp x = Operation mul [Constant (negate 1.0), x]
subOp (a:b:_) = Operation add [a, negOp b]-}

estimate :: Operation Float -> M.Map String Float -> Float
estimate op vars = calc op
    where
        calc (Constant c) = c
        calc (Variable v) = case M.lookup v vars of 
            Just x -> x
            Nothing -> error "bad var name"
        calc (Operation f args) = calcFunction f $ map calc args
        
        calcFunction (Function s f)  args = f args
        calcFunction (BinaryOperator s o) (a:b:_) = a `o` b
        calcFunction (UnaryOperator s o) (a:_)    = o a

isConstant :: Operation a -> Bool
isConstant (Constant _)    = True
isConstant (Variable _)    = False
isConstant (Operation _ l) = all isConstant l

isConstantOf :: String -> Operation a -> Bool
isConstantOf _ (Constant _)    = True
isConstantOf m (Variable x)    = m /= x
isConstantOf m (Operation _ l) = all (isConstantOf m) l

isVarOf v = not . isConstantOf v

simplifyConst :: Operation Float -> Operation Float
simplifyConst (Operation f l) 
    | all isConstant l = Constant $ estimate (Operation f l) (M.empty)
    | otherwise = Operation f $ map simplifyConst l
simplifyConst v = v

simplifyExpr :: Operation Float -> Operation Float
simplifyExpr (Operation (BinaryOperator op _) (a:b:_) ) = simplifyBinaryOperator op
    where
        simplifyBinaryOperator :: String -> Operation Float
        simplifyBinaryOperator "+" 
            | a' == Constant 0 = b'
            | b' == Constant 0 = a'
            | otherwise = Operation add [a', b']
        simplifyBinaryOperator "^" 
            | b' == Constant 1 = a' 
            | b' == Constant 0 = Constant 1
            | a' == Constant 0 = Constant 0
            | otherwise = Operation pow [a', b']
        simplifyBinaryOperator "*" 
            | (a' == Constant 0) || (b' == Constant 0) = Constant 0
            | b' == Constant 1 = a'
            | a' == Constant 1 = b'
            | otherwise = Operation mul [a', b']
        a' = simplifyExpr a
        b' = simplifyExpr b 

simplifyExpr (Operation f l) = Operation f $ map simplifyExpr l
simplifyExpr x = x

simplify = simplifyExpr . simplifyConst

funcDerr :: String -> Function Float -> [Operation Float] -> Operation Float
funcDerr v (BinaryOperator "+" _) l = Operation add $ map (partialDerr v) l
funcDerr v (BinaryOperator "*" _) (a:b:_) = Operation add [
                    Operation mul [partialDerr v a, b],
                    Operation mul [a, partialDerr v b]
                ]
funcDerr v (BinaryOperator "^" _) (a:b:_) 
    | (isConstantOf v b) && (not $ isConstantOf v a) = Operation mul [b, Operation pow [a, subOp [b, (Constant 1)]]]
    | isConstantOf v a = Constant 0
    | otherwise = error "NO" 

funcDerr v func args = Operation (UnaryOperator ("d/d" ++ v) id) [Operation func args]

partialDerr :: String -> Operation Float -> Operation Float
partialDerr var (Constant _) = Constant 0
partialDerr var (Variable x) 
    | x == var  = Constant 1
    | otherwise = Constant 0
partialDerr var (Operation f args) = funcDerr var f args

partialDerr2 v = partialDerr v . partialDerr v
