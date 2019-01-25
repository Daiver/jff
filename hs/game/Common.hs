module Common where

import Control.Arrow

mapTuple :: (a -> b) -> (a, a) -> (b, b)
mapTuple f = f *** f

updateListBy :: (a -> a) -> Int -> [a] -> [a]
updateListBy f 0 (x:xs) = f x : xs
updateListBy f n (x:xs) = x : updateListBy f (n - 1) xs

updateList v = updateListBy (const v)

updateList2DBy :: (a -> a) -> (Int, Int) -> [[a]] -> [[a]]
updateList2DBy f (x, y) = updateListBy (updateListBy f y) x 

updateList2D v = updateList2DBy (const v)
