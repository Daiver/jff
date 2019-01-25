module Main where

import GA.Simple
import System.Random
import Text.Printf
import Data.List as L
import Control.DeepSeq

type Vector = [Double]
type Matrix = [Vector]
type Tenzor = [Matrix]

accum f g l x = l `f` g x

mulDbyV :: Double -> Vector -> Double
mulDbyV x = foldl' (accum (+) (*x)) 0

mulVbyV :: Vector -> Vector -> Double
--mulVbyV v1 = foldl' (accum (+) (flip mulDbyV v1)) 0 . reverse
mulVbyV v1 v2 = sum . map (uncurry (*)) $ zip v1 v2
mulVbyM v1 = map (mulVbyV v1) . L.transpose
mulMbyM m1 m2 = map (flip mulVbyM m2) m1

changeElem m x y v = map (\i -> map (work i) [0..h - 1]) [0.. w - 1]
    where 
        w = length m
        h = length (head m)
        work i j 
            | i == x && j == y = v
            | otherwise = m !! i !! j

eachElemVbyV = zipWith 
eachElemMbyM f = zipWith (eachElemVbyV f) 
eachElemM f = map (map f)

sigmoid x = 1.0/(1 + exp (negate x))

activate :: Vector -> Matrix -> Vector
activate v m = map sigmoid . mulVbyM v $ (L.transpose m)

activateTens :: Vector -> [Matrix] -> Vector
activateTens v w = f v w
    where 
        f v [] = v
        f v w = f (activate . head $ w) (tail w)

--field 10 x 10
samples = [
    [1,9],
    [2,7],
    [2,9],
    [5,4],
    [9,3],
    [6,5]
    ]

labels = [
    [0],[0],[0],[1],[1],[1]
    ]

test_samples = [[3,2],[5,9]]
test_labels = [1,0]

data NNWeights = NNWeights [Matrix] deriving Show

instance NFData NNWeights where
    rnf (NNWeights xs) = rnf xs `seq` ()

mutationOne g xs = --(NNWeights xs, g)
    let (idx, g') = randomR (0, length xs - 1) g
        (idy, g'') = randomR (0, length (head xs) - 1) g'
        (dx, g''') = randomR (-10.0, 10.0) g'
        xs' = changeElem xs idx idy dx
    in (xs', g''')


instance Chromosome NNWeights where
    crossover g (NNWeights xs) (NNWeights ys) =
        ( [ NNWeights $ map (uncurry (eachElemMbyM (\x y -> (x+y)/2))) $ zip xs ys], g)

    mutation g (NNWeights xs) = (NNWeights xs', g')
        where
            (xs', g') = L.foldl'
                

    fitness int =
        let max_err = 1000.0 in
        max_err - (min (err labels samples int) max_err)


err :: Matrix -> Matrix -> NNWeights -> Double
err labels samples (NNWeights w) = 
    sum . concat . eachElemMbyM (\x y -> abs (x - y)) labels $ 
        map (flip activate w) samples

randomGener gen = 
    let (lst, gen') =
            L.foldl'
                (\(xs, g) _ -> let (x, g') = randomR (-20.0,20.0) g in (x:xs,g') )
                ([], gen) [0..1]
    in (NNWeights [lst], gen')

stopf :: NNWeights -> Int -> IO Bool
stopf best gnum = do
    let e = err labels samples best
    _ <- printf "Generation: %02d, Error: %.8f\n" gnum e
    return $ e < 0.0002 || gnum > 50


main = do
    {-int <- runGAIO 64 0.3 randomSinInt stopf
    putStrLn ""
    putStrLn $ "Result: " ++ show int-}
    print "Start"
    print $ mulDbyV 10 [1,2,3]
    print $ mulVbyV [3,2,1] [1,2,3]
    print $ mulVbyM [3,2,1] [[1,2,3],[4,5,6], [7,8,9]]
    print $ mulMbyM [[1,2,3],[4,5,6],[7,8,9]] [[1,2,3],[4,5,6], [7,8,9]]
    print $ mulMbyM [[1,1,1,1],[1,1,1,1],[1,1,1,1]] [[1,2,3],[4,5,6], [7,8,9],[1,1,1]]
    print $ changeElem [[1,2,3, 1],[4,5,6, 1], [7,8,9, 1]] 2 1 1000
    print $ eachElemMbyM (+) [[1,1,1],[1,1,1],[1,1,1]] [[1,2,3],[4,5,6], [7,8,9]]
    print $ activate [10,10,10] [[0,1], [1,2], [2,3]]
    print $ err 
                [[2,4], [0,0]] 
                [
                    [1,0,1], 
                    [10,10,10]
                ] 
                (NNWeights [[0,1], [1,2], [2,3]])
    int@(NNWeights mm) <- runGAIO 64 0.9 randomGener stopf
    putStrLn ""
    putStrLn $ "Result: " ++ show int
    print $ activate (head test_samples) mm
    print $ activate (last test_samples) mm
    print "Finish"
