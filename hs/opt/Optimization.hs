module Optimization where

import Control.Monad
import Control.Monad.Writer

gradientDescent, gradientDescent2 :: (Num a, Ord a, Fractional a) =>
                   ([a] -> [a]) -> 
                   ([a] -> a) -> 
                   a -> [a] -> 
                   Writer [String] [a]

gradientDescent grad costF thr initArg 
    | abs err < thr = return initArg
    | otherwise           = (gradientDescent grad costF thr $ newInitArg) 
    where
        stepConst = 0.01
        err = costF initArg - costF newInitArg
        newInitArg = zipWith (-) initArg . map (*stepConst) $ grad initArg

gradientDescent2 grad costF thr initArg = inner 100000 (grad initArg) initArg
    where
        stepConst = 0.01
        inner err lastGrad initArg 
            | abs err < thr         = return initArg
            | abs err > abs nextErr = inner nextErr lastGrad initArg
            | otherwise             = inner newErr newGrad newInitArg 
            where
                err = costF initArg - costF newInitArg
                nextErr = costF nextInitArg
                newErr = costF newInitArg
                nextInitArg = zipWith (-) initArg . map (*stepConst) $ lastGrad
                newGrad = grad initArg
                newInitArg = zipWith (-) initArg . map (*stepConst) $ newGrad


update :: Int -> (a -> a) -> [a] -> [a]
update ind v li = map f $ zip li [0..]
    where
        f (x, i) 
            | i == ind = v x
            | otherwise = x

numericalDer :: (Fractional a) => ([a] -> a) -> Int -> [a] -> a
numericalDer f ind arg = (f (update ind (+const) arg)- f arg) / const
    where const = 0.01

numericalGrad :: (Fractional a) => ([a] -> a) -> [a] ->  [a]
numericalGrad f arg = map (flip (numericalDer f) arg) [0.. length arg - 1]
