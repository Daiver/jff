import Control.Applicative
import Control.Monad

import System.Environment
import Control.Monad.IO.Class (liftIO)
import Control.Monad
import Data.AStar
import Data.Common

data FieldState = FieldState (Int, Int) deriving (Show, Eq, Ord)

expandField field (FieldState (x,y)) = filter isWalkable $  map addStep [(1,0),(0,1),(-1,0),(0,-1)]
    where 
        addStep (nx, ny) = FieldState (nx + x, ny + y)
        isWalkable (FieldState (x, y)) = (x > -1 && y > -1 && y < length field && x < length (field !! 0)) && (field !! y !! x /= "1")

fieldH (gx, gy) (FieldState (x,y)) = negate (abs(gx - x) + abs(gy - y))
fieldFinish (gx, gy) (FieldState (x,y)) = (gx == x) && (gy == y)

readLabFromString s = (field, goalPos, startPos)
    where
        field = map words $ lines s
        w = length (field !! 0)
        h = length field
        goalPos = head . find2dChar $ "X"
        startPos = FieldState $ head . find2dChar $ "S"
        find2dChar char = do
            i1 <- [0..h-1]
            i2 <- [0..w-1]
            if (field !! i1 !! i2) == char
                then return (i2, i1)
            else []

main = do
    content <- (readFile . head =<< getArgs)
    let (field, goal, start) = readLabFromString content
    print $ (field, goal, start)
    print $ join $ map (expandField field) $ expandField field (FieldState (1,4))
    print $ map (fieldH goal) $ join $ map (expandField field) $ expandField field start
    let (res, p, o) = aStar (fieldFinish goal) (\ _ _ -> 1) (fieldH goal) (expandField field) start 
    print res
    print $ map (\x -> (value x, distance x)) o
    print $ map (fieldH goal) $ map value o
