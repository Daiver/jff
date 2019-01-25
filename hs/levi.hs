
levi :: String -> String -> Int
levi s1 s2 = d (length s1) (length s2)
    where
        d :: Int -> Int -> Int
        d 0 0 = 0
        d i 0 = i
        d 0 j = j
        d i j = minimum [
                d i (j - 1) + 1,
                d (i - 1) j + 1,
                d (i - 1) (j - 1) + m (s1 !! (i - 1)) (s2 !! (j - 1))
            ]

        m c1 c2 
            | c1 == c2 = 0 
            | otherwise = 1 

main = do
    print "Start"
    print $ levi "cash" "clash"
    print $ levi "mx" "maximum"
    print $ levi "mx" "maxim"
