import Data.List
import System.Random
-- Number of samples to take
count = 100000

-- Function to process our random sequence
process :: [(Double, Double)] -> (Int, Int)
process = foldl sumInCircle (0, 0)

-- Function to process a running value and a random value, producing a new running value.
sumInCircle :: (Int, Int) -> (Double, Double) -> (Int, Int)
sumInCircle (ins, total) (x, y) = (ins + if x*x + y*y < 1.0 then 1 else 0,
                                   total + 1)

-- Function to display the running value.
display:: (Int, Int) -> String
display (heads, coins) = "π = " ++ (show $ 4.0 * fromIntegral heads / fromIntegral coins)

-- function to prepare the random sequence for processing
prep :: [Double] -> [(Double, Double)]
prep (a:b:r) = (a,b):prep r

prep2 :: [Int] -> [Int]
prep2 (a:r) = (abs a):(prep2 r)

prep3 :: [Int] -> [(Int, Int)]
prep3 (a:b:r) = let a' = if a `mod` 2 == 0 then a + 1 else 1
        in (a', b `mod` a'):prep3 r

align a b x = a + (x `mod` b)

specTest :: [Int] -> [(Bool, (Int, Int))]
specTest (n:a:xs) -- = (1,1)
        | n `mod` a /= 0 = (False, (n', a')) : specTest xs
        | testA     = [(True, (n', a'))]
        | otherwise = (False, (n', a')) : specTest (n:xs)
    where
        n' = if n `mod` 2 == 0 then n + 1 else n
        a' = align 0 n' a
        testA = odd a'

main = do
    g <- newStdGen 
    g1 <- newStdGen 
    print $ snd . head . filter fst . specTest $ randomRs (1000,100000) g 
    --print $ takeWhile (>7) . prep2 $ randoms g 
    {-print $ take 10 . prep2 $ randoms g 
    print $ take 10 . prep2 $ randoms g 
    print $ take 10 . prep2 $ randoms g1
    print $ take 10 . prep2 $ randoms g1
    print $ take 10 . prep2 $ randoms g
    a <- randomRIO (1, 10) :: IO Int
    b <- randomRIO (1, 10) :: IO Int
    c <- randomRIO (1, 10) :: IO Int
    print (a, b, c)
    --putStrLn . display .process . take count . prep $ randoms g -}

