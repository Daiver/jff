import Data.Char
import Data.SimpleJSON
import Data.Common
import qualified Data.Map as M
import Control.Monad.Writer
import Control.Monad.State
import Control.Monad.Reader

disp :: [Float] -> Float
disp seq = let sm = sum seq
               mean = sm / (fromIntegral . length $ seq)
             in (sum . map (\x -> abs (x - mean)) $ seq) / (fromIntegral . length $ seq)

caseTest :: Maybe a -> a -> a
caseTest value defvalue = 
    case value of
        Nothing -> defvalue
        Just x  -> x

myLength :: [a] -> Int
myLength x = getLength x 0
    where 
        getLength (x:xs) n = getLength xs (n + 1)
        getLength [] n = n

intersperce :: a -> [[a]] -> [a]
intersperce sep seq = foldl1 (\lst x -> lst ++ [sep] ++ x) seq

data Record = Record {time::String, power::Integer} deriving (Show, Read)

readRecord:: [String] -> (String, [Record])
readRecord (mac:time:power:_) = (mac, [Record time (read power)])

formSet simple_data = M.fromListWith (++) [x | x <- simple_data]

logNumber :: Int -> Writer [String] Int  
logNumber x = writer (x, ["Got number: " ++ show x]) 

getToken :: State  (String, Int) String
getToken = do
    (src, pos) <- get
    put (src, pos + 1)
    return "1"

main = do
    --print $ runState  getToken ("10000", 0)
    print $ liftM2 (*) (Just 5 ) (Just 10)
    --print $ runWriter $ logNumber 100
    --raw_data <- readFile "dump1"
    --let simple_data = map (readRecord . words) $ lines raw_data
    --print $ formSet simple_data
    --print $ readL "[1, 2, 3]"
    {--print "HI"
    print $ caseTest Nothing 0
    print $ caseTest (Just "NOT") ""
    print $ disp [1,1 ,1, 1]
    print $ myLength [1,1 ,1, 1]
    print $ intersperce ' ' ["1", "2", "3", "4"]
    --}
    
