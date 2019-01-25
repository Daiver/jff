import Control.Applicative
import Control.Monad
import qualified Data.List as L
import qualified Data.Set as S
import qualified Data.List.Ordered as O
import Data.Maybe
import Data.Ord

import Data.AStar

data FieldState = FieldState ((Int, Int), (Int, Int)) deriving (Show, Eq, Ord)

expandField field (FieldState (goal,(x,y))) = map addStep [(1,0),(0,1),(-1,0),(0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]
    where addStep (nx, ny) = FieldState (goal, (nx + x, ny + y))

testField = FieldState ((2,2), (0,0))
testMap = [
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ]

fieldH (FieldState ((gx, gy), (x,y))) = negate (abs(gx - x) + abs(gy - y))

fieldFinish (FieldState ((gx, gy), (x,y))) = (gx == x) && (gy == y)

main = do
    print "Start"
    let (v, p,res) = aStar fieldFinish (\ _ _ -> 1) fieldH (expandField testMap) testField 
    print $ map value res
    print p
