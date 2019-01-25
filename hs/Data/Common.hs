module Data.Common where

import Control.Applicative
import qualified Data.Map as M
import qualified Data.List as L
import qualified Data.Set as S
import Control.Monad
import Data.List.Split(splitOn)
import Data.Function
import Data.Maybe
import qualified Data.Char as C
--import qualified Data.String.Utils as SU

--massReplace candidats str = foldl (\ s c -> SU.replace (fst c) (snd c) s) str candidats

enumerate x = [0..(length x) - 1]
decart x y = [(a, b) | a <- x, b <- y]
uniquify l = S.toList . S.fromList $ l

toLower :: [Char] -> [Char]
toLower s = map C.toLower s

freqDict input = M.fromListWith (+) [(c, 1) | c <- input]
normFreqDict input = M.map (/(fromIntegral . length $ input)) 
freqDictF input = normFreqDict input . freqDict $ input

toSimpleList = foldl (++) []

readInt::[Char] -> Int
readInt s = read s
--
average :: [Int] -> Float
average lst = (fromIntegral . sum $ lst) / (fromIntegral . length $ lst)
median :: [Int] -> Float
median lst = fromIntegral $ L.sort lst !! (length lst `div` 2)

partitionBy f = map snd . M.toList . M.fromListWith (++) . map (\x -> (f x, [x]))

justOrValue _ (Just x)  = x
justOrValue val Nothing = val
