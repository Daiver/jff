import Control.Applicative
import qualified Data.Map as M
import qualified Data.List as L
import qualified Data.Set as S
import qualified Data.String.Utils as SU
import Control.Monad
import Control.Arrow
import Data.List.Split(splitOn)
import Data.Function

import Data.DecisionTree
import Data.Common

normilizeData s = massReplace delim_list . toLower $ s
    where 
        delim = ",.!@#$%^&()-=+_[]{}'\"/\\|`~:;?<>"
        delim_list = map (\x -> ([x], [' ', x, ' '])) delim

--featsFromList :: [String] -> [String] -> [Integer]
featsFromList features sl = foldl (\l x -> mergeMaybe l  $ M.lookup x freq) [] features
    where 
        freq = freqDict sl
        mergeMaybe l (Just x) = l ++ [x]
        mergeMaybe l Nothing = l ++ [0]

--readSpamData :: [Char] -> ([String], [(Integer, [Integer])])
readSpamData str =  (features, samples) --map (filter (/= "") . (splitOn " ") $ lines
    where 
        raw = normilizeData str 
        lines = splitOn "\n" raw
        splited_data = map (\x -> (if "ham" == (head x) then 1 else 0, tail x))
            . filter (\x -> length x > 0) . map (filter (/= "") . splitOn " ") $ lines
        freq = freqDict . toSimpleList $ map snd splited_data
        features = M.keys freq
        samples = map (\(x, y) -> (x, featsFromList features y)) splited_data

main 
    = do
        print $ "start"
        print . length . fst . readSpamData =<< readFile "spamdata" --"smallspam"
        print $ "end"
