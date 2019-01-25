import Control.Arrow

tupleMap f = f *** f

merge [] ys = ys
merge xs [] = xs
merge xa@(x:xs) ya@(y:ys)
    | x < y = x : merge xs ya
    | otherwise = y : merge xa ys

mergeSort [] = []
mergeSort [x] = [x]
mergeSort li = uncurry merge . tupleMap mergeSort . splitAt (length li `div` 2) $ li

main = do
    print $ mergeSort [4,3,6,1,5]
