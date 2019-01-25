module ML.Common where

import Control.Arrow
import Control.Monad
import qualified Data.List as List
import qualified Data.Map as Map
import System.Random
import Control.Monad.ST
import Data.STRef
import Data.Array.ST


item = flip (!!)
mapTuple x = join (***) x

counts :: (Ord a) => [a] -> [(a, Int)]
counts input = Map.toList $ Map.fromListWith (+) [(c, 1) | c <- input]

freqs :: [Int] -> [Float]
freqs cnts = map ((/s) . fromIntegral) cnts
    where s = fromIntegral . sum $ cnts

maximumWith f = List.maximumBy (\x y -> compare (f x) (f y))

class Classifier a where
    predict :: a -> [Float] -> Int
    fit     :: [(Int, [Float])] -> a

testClassifier :: (Classifier c) => c -> [(Int, [Float])] -> Int
testClassifier classifier testData = length . filter not $ zipWith (\smlp lbl -> predict classifier smlp == lbl) samples labels
    where
        labels  = map fst testData
        samples = map snd testData

reasSamplesClassIsLast :: String -> [(Int, [Float])]
reasSamplesClassIsLast = map sampleFromLine . wrds
    where
        wrds = map words . lines
        sampleFromLine line = (read $ last line, map read $ init line)

reasSamplesClassIsLead :: String -> [(Int, [Float])]
reasSamplesClassIsLead = map sampleFromLine . wrds
    where
        wrds = map words . lines
        sampleFromLine line = (specRead $ head line, map specRead $ tail line)
        specRead ('.' : xs) = read $ "0." ++ xs
        specRead x = read x

randomSeq :: StdGen -> Int -> ([Int], StdGen)
randomSeq gen len = (take len . randoms $ gen', gen'')
    where
        (gen', gen'') = split gen

randomSeqInRange gen len a b = 
    let (seq, gen'') = randomSeq gen len in (map (\x -> a + (x `mod` (b - a))) $ seq, gen'')

shuffle :: [a] -> StdGen -> ([a],StdGen)
shuffle xs gen = runST (do
        g <- newSTRef gen
        let randomRST lohi = do
              (a,s') <- liftM (randomR lohi) (readSTRef g)
              writeSTRef g s'
              return a
        ar <- newArray n xs
        xs' <- forM [1..n] $ \i -> do
                j <- randomRST (i,n)
                vi <- readArray ar i
                vj <- readArray ar j
                writeArray ar j vi
                return vj
        gen' <- readSTRef g
        return (xs',gen'))
  where
    n = length xs
    newArray :: Int -> [a] -> ST s (STArray s Int a)
    newArray n xs =  newListArray (1,n) xs

subSample li = map ((!!) li) 

randomSampling gen li = (subSample li indxs, gen')
    where (indxs, gen') = shuffle [0..length li - 1] gen

randomSamplingWithRepeats gen li = (subSample li indxs , gen')
    where (indxs, gen') = randomSeqInRange gen (length li) 0 (length li - 1)
