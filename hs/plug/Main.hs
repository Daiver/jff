module Main (main) where

import System.Plugins

main :: IO ()
main = do
    putStrLn "Loading"
    mv <- dynload "Plug.o" [] [] "thing2"   -- also try 'load' here
    putStrLn "Loaded"
    case mv of
        LoadFailure msgs -> putStrLn "fail" >> print msgs
        LoadSuccess _ v -> do
            putStrLn "success"
            print $ (v::(Int -> Int)) 100
