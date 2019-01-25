
compose :: [a -> a] -> (a -> a)
compose = foldr (.) id

main = print $ compose [sum] [1,23]
