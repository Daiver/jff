module ML.DecisionTree where

import System.Random
import qualified Data.List as List
import qualified Data.Map as Map
import Control.Monad
import Control.Arrow

import ML.Common

data DecisionTree = Node Int Float (DecisionTree) (DecisionTree) | Leaf [(Int, Int)]
    deriving Show

instance Classifier DecisionTree where
    predict tree feats = predictDecTree tree feats
    fit samples = buildNode samples

commonGain :: ([Int] -> Float) -> [Int] -> [Int] -> [Int] -> Float
commonGain h s s1 s2 = h s - (l1/l) * h s1 - (l2/l) * h s2
    where 
        l  = fromIntegral $ length s
        l1 = fromIntegral $ length s1
        l2 = fromIntegral $ length s2

gini :: [Float] -> Float
gini fr = 1 - (sum . map (**2) $ fr)

divide samples featIdx thr = List.partition ((>= thr) . item featIdx . snd) samples

findBestThrBruteForce featIndexes samples = List.maximumBy (\(x,_,_) (y,_,_) -> compare x y) genProps
    where
        countOfFeats = length . snd $ (samples !! 0 )
        h = gini . freqs . map snd . counts
        labels = map fst samples
        initGain = h labels
        pairs = [(i, (snd s) !! i) | i <- featIndexes, s <- samples]
        computeGainOfDivide i thr = uncurry (commonGain h labels) . mapTuple (map fst) $ divide samples i thr
        genProps = map (\(i, thr) -> (computeGainOfDivide i thr, i, thr)) pairs

buildNode :: [(Int, [Float])] -> DecisionTree
buildNode samples
    | bestGain > 0 = let (r, l) = divide samples featIdx thr in Node featIdx thr (buildNode l) (buildNode r)
    | otherwise = Leaf . counts . map fst $ samples
    where 
        (bestGain, featIdx, thr) = findBestThrBruteForce [0..length (snd $ samples !! 0) - 1] samples

predictDecTree :: DecisionTree  -> [Float] -> Int
predictDecTree (Leaf vals) _ = fst . List.maximumBy (\x y -> compare (snd x) (snd y)) $ vals
predictDecTree (Node featIdx thr l r) feats
    | feats !! featIdx >= thr = predict r feats
    | otherwise               = predict l feats

