

import Data.Common
import Data.DecisionTree
import Data.List.Split(splitOn)


data Useless a = ULess a | UNothing
data Car = Car {company :: String, model :: String, year :: Int} deriving (Show)  

data Vector a = Vector a a a deriving (Show)

vplus :: (Num a) => Vector a -> Vector a -> Vector a
vplus (Vector a1 b1 c1) (Vector a2 b2 c2) = Vector (a1 + a2) (b1 + b2) (c1 + c2)

data RNode = ANode | BNode deriving (Show, Eq)
data Road = Road [RNode] Int

roadsWork = do
    raw_data <- readFile "roads1"
    let lns = map readInt . filter (/= "") . splitOn "\n" $ raw_data
    let roads = makeTriple lns
    print roads
    where 
        makeTriple [] = []
        makeTriple (a:b:c:xs) = [a, b, c] : makeTriple xs
        buildAll cur (a:b:c:_) = 1--map addNode ANode

feats = ["x", "y"]
circleRect n = [
            (ifInCircle x y, [x, y])
            | x <- [-n..n], y <- [-n..n]
        ]
        where ifInCircle x y 
                | (x*x + y*y) < n*n = 1
                | otherwise = 0

main = do
    let data_c = circleRect 15
    let tree = makeSimpleDecTree ginii (>) feats data_c
    print $ tree
    print $ sum $ map (\x -> (maxT $ computeDecTree tree $ (snd x))) data_c
    print $ sum $ map (\x -> invs . fst $ x) data_c
    where 
        maxT (x, y) 
                    | x > y = 0
                    | otherwise = 1
        invs 1 = 0
        invs 0 = 1
--main = testAlg =<< readFile "tmp"
--print $ (Vector 1 2 3) `vplus` (Vector 4 5 6)
