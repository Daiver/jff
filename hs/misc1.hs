import System.Random  
import Control.Monad
import Control.Monad.State  
import Data.Time.Clock

randomSt :: (RandomGen g, Random a) => State g a  
randomSt = state random  

threeCoins :: State StdGen [Int]
threeCoins = do  
    d <- sequence . replicate 10 $ randomSt
    let t = map (`mod` 100) d
    return t

randTest = do
    cur_time <- getCurrentTime
    let timed = floor $ utctDayTime cur_time :: Int
    print $ runState threeCoins $ mkStdGen timed
    print "hi"

integral f a b num_of_steps = (* step) . foldl1 (\s x -> s + f x) $ [a, step..b]
    where step = (b - a) / num_of_steps

main = do
    print $ integral ((**2) . sin) 0 100 1000000
