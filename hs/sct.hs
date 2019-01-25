{-# LANGUAGE OverloadedStrings #-}
import Web.Scotty

import Data.Monoid (mconcat)
import System.Random (newStdGen, randomRs)

main :: IO ()
main = scotty 3000 $ do
    get "/:word" $ do
        beam <- param "word"
        html $ mconcat ["<h1>Scotty, ", beam, " me up!</h1>"]
    get "/ints/:is" $ do
        --is <- param "is"
        json $ ("123" :: String) -- [(1::Int)..10] ++ is

