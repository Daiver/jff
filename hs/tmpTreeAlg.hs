import Control.Monad
import Control.Monad.ST
import Control.Arrow
import qualified Data.Map as Map
import qualified Data.List as List

counts :: (Ord a) => [a] -> [(a, Int)]
counts input = Map.toList $ Map.fromListWith (+) [(c, 1) | c <- input]

updateElem :: (a -> a) -> [a] -> Int -> [a]
updateElem f li i = 
    let(bef, aft) = splitAt i li in concat $ [bef, [f $ head aft], tail aft]

findBestThr :: [(Int, [Float])] -> ([Int], [Int], Float, Int, Float)
findBestThr samples = 
        foldl (foldlCore labels dt initGain) 
              (initWl, initWr, dt !! 0 !! 0, 0, initGain) $ indexesAndFeats
    where 
        labels = map fst samples
        dt     = map snd samples
        initWr = map snd $ counts labels
        initWl = replicate (length initWr) 0
        initGain = giniFromCounts . map snd . counts $ labels
        countOfFeats = length $ dt !! 0
        indexesAndFeats = concat $ map 
                            (\f ->map ((,) f) $ List.sortBy 
                                (\a b -> compare (dt!!a!!f) (dt!!b!!f)) 
                                [0.. length dt - 1]) 
                            [0 .. countOfFeats - 1]


foldlCore labels dt initGain state@(wl, wr, bestThr, bestFeat, bestGain) (idx, feat)
    | gain > bestGain = (wl', wr', thr, feat, gain)
    | otherwise = (wl', wr', bestThr, bestFeat, bestGain)
    where
        l   = fromIntegral $ length labels
        l1  = fromIntegral $ sum wl
        l2  = fromIntegral $ sum wr
        label = labels !! idx
        thr = dt !! idx !! feat
        wl' = updateElem (+1) wl label
        wr' = updateElem (\x -> x - 1) wr label
        gain = initGain - giniFromCounts wl' * l1/l + giniFromCounts wr' * (l2/l)

giniFromCounts :: [Int] -> Float
giniFromCounts counts = 
    1 - (fromIntegral (sum $ map (^2) counts) / fromIntegral (length counts ^ 2))

main = do
    let a = [1,2,3,4,5,6,7,8,9]
    print $ updateElem (+1) a 0
    print $ updateElem (+1) a 1
    print $ updateElem (+1) a 2
    print $ updateElem (+1) a 8
    {-print "Gen"
    let dt = concat $ [[1..10] | _ <- [1..100000]]
    print "Start"
    print $ counts dt-}
