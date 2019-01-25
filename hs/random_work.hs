
import Data.Common
import Data.DecisionTree
import Data.List.Split(splitOn)
import System.Random

randomIn :: Int -> Int -> IO Int
randomIn a b = getStdRandom (randomR (a, b))

--xs = sequence . replicate 10 $ rollDice

randomStuff :: RandomGen g => g -> [Float]
randomStuff g = take 2 (randomRs (0.0, 1.0) g)

randomStuff2 :: RandomGen g => g -> [[Float]]
randomStuff2 g = (randomRs (0.0, 1.0) g)

work :: Int -> [Float] -> [[Float]]
work x (r:rs) =
    let n = truncate (r * 7.0) + x
        (xs,ys) = splitAt n rs
    in xs : work x ys

main = do
    print "Type random base"
    a <- getLine
    let b = read a
    let g = mkStdGen b
    print $ take 10 $ randomStuff2 g
    --print $ take 10 $ randomStuff g
