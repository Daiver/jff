
import Data.Common
import Data.List.Split(splitOn)
import System.Random
import Data.Time.Clock

randomIn :: Int -> Int -> IO Int
randomIn a b = getStdRandom (randomR (a, b))

--xs = sequence . replicate 10 $ rollDice

randomStuff :: RandomGen g => g -> [Float]
randomStuff g = take 2 (randomRs (0.0, 1.0) g)

work :: Int -> [Float] -> [[Float]]
work x (r:rs) =
    let n = truncate (r * 7.0) + x
        (xs,ys) = splitAt n rs
    in xs : work x ys

genAlg random_gen init_pop = init_pop

mkInitRandomList :: (RandomGen a, Num t, Random t) => a -> Int -> t -> t -> [t]
mkInitRandomList gen n a b = take n $ randomRs (a, b) gen

getA :: Int
getA = 1

getB :: Int
getB = 10

main = do
    currTime <- getCurrentTime
    let timed = floor $ utctDayTime currTime :: Int
    let gens = replicate 10 $ mkStdGen timed
    print $ mkInitRandomList (head gens) 10 getA getB
    print $ mkInitRandomList (head (tail gens)) 10 getA getB
