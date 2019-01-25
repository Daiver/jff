import System.Random

monteCarlo :: (RandomGen g) => (Float -> Float) -> Float -> Float -> Int -> g -> Float
monteCarlo f a b n g = 
    let 
        values = take n $ randomRs (a, b) g 
        sm     = sum . map f $ values
        mean   = sm / (fromIntegral n)
    in (b - a) * mean

main = do
    print "Hi"
    g <- newStdGen
    print $ monteCarlo (\x -> 10.0 - x * x) 0.0 2.0 100000 g
