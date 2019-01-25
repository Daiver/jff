module Main where

-- Import our template "pr"
import Printf ( pr, parse, fact )

-- The splice operator $ takes the Haskell source code
-- generated at compile time by "pr" and splices it into
-- the argument of "putStrLn".

main = do
    print $ parse "hello %d"
    putStrLn ( $(pr "Hello %d %s %d") 1 "oh" [2])
    print $(fact 5)
