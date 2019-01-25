module Data.DecisionTree where

import Control.Applicative
import qualified Data.Map as M
import qualified Data.List as L
import qualified Data.Set as S
import Control.Monad
import Control.Arrow
import Data.List.Split(splitOn)
import Data.Function

import Data.Common

data DecPredicate a = DecPredicate {test::a->Bool, text::String} 
instance Show (DecPredicate a) where
    show (DecPredicate test text) = "{" ++ show text ++ "}"

data DecTree a b = DecLeaf b | DecNode (DecPredicate a) (DecTree a b) (DecTree a b) deriving Show


computeDecTree (DecLeaf x) _ = x
computeDecTree (DecNode p l r) env
                            | test p env = computeDecTree r env
                            | otherwise = computeDecTree l env

divSamples divfunc samples feat_num val = L.partition (\ x -> (snd x)!!feat_num `divfunc` val) samples

uniqResults samples = (length . fst $ parts, length . snd $ parts)
            where parts =  L.partition ((==1) . fst) samples

ginii samples = (fromIntegral ((fst ur) * (snd ur))) / ((**2) . fromIntegral . length $ samples)
            where ur = uniqResults samples

makeSimpleDecTree compute_gain divfunc features samples = makeSubDecTree samples
    where
        feat_nums = enumerate features
        feat_values = map uniquify . L.transpose . map snd $ samples
        makeSubDecTree samples 
                            | max_gain > 0 = DecNode 
                                            (DecPredicate (\ x -> x!!max_feat `divfunc` max_feat_val) ((show max_feat) ++ "-" ++ (show max_feat_val)))
                                            (makeSubDecTree (snd max_div)) 
                                            (makeSubDecTree (fst max_div))
                            | otherwise = DecLeaf ur
            where 
                ur = uniqResults samples
                cur_gain = compute_gain samples
                gain_from_feat feat_num feat_val = (cur_gain 
                                            - p * (compute_gain . fst $ divs) 
                                            - (1-p)*(compute_gain . snd $ divs), divs, feat_num, feat_val)
                    where 
                        divs = divSamples divfunc samples feat_num feat_val
                        p = (fromIntegral . length . fst $ divs) /  (fromIntegral . length $ samples)
                (max_gain, max_div, max_feat, max_feat_val) = maximum . map (\ x -> maximum . map (\ y -> gain_from_feat x y) $ feat_values !! x) $ feat_nums


--readInt::[Char] -> Int
--readInt s = read s

getDataFromFile str = (feats, samples)
    where
        lines = splitOn "\n" str
        feats = splitOn " " . head $ lines
        mat = [
                    [ x | x <- reverse l]  
                    | l <- map (splitOn " ") . filter (/="") . tail $ lines
                ]
        samples = map (\x -> (readInt . head $ x, reverse . tail $ x)) mat

testAlg str = do
        print tree 
        print $ computeDecTree tree ["-5", "1", "1", "0"]
    where
        tree = makeSimpleDecTree ginii (>) features samples
        (features, samples) = getDataFromFile str
{-
readSpamData str = splited_data
    where 
        
        lines = splitOn "\n" str
        splited_data = map (\x -> (if "ham" == head x then 1 else 0,
                tail x))
            . map (splitOn " ") $ lines

main 
    = do
        print $ "start"
        print . readSpamData =<< readFile "smallspam"
        --testAlg =<< readFile "tmp"

--let feat_values = map uniquify . transpose . map snd $ samples
--uniqResults samples
--divSamples samples 1 1
--computeDecTre
--e tree 1
--print feat_values
--print . getDataFromFile =<< readFile "tmp"
--let tree = makeSimpleDecTree ginii features samples
--print $ tree
--print $ computeDecTree tree [2, 0, 0]

-}

