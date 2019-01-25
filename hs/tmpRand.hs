import System.Random

myPoorFunc seed = [val1, val2, val3]
    where 
        genInit    = mkStdGen seed
        (val1, g1) = random genInit :: (Int, StdGen)
        (val2, g2) = random g1 :: (Int, StdGen)
        (val3, g3) = random g2 :: (Int, StdGen)

main = print $  myPoorFunc 42

