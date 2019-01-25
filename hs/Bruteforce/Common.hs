module Bruteforce.Common where

import Data.Word
import Text.Printf
import Data.Digest.SHA2

passwordList :: String -> Int -> [String]
passwordList charList len = 
    stream beginState
  where
    beginState = replicate len charList
    endState = replicate len [ last charList ]
    nextState ((_:[]):xs) = charList : nextState xs
    nextState ((_:ys):xs) = ys : xs
    nextState x = error $ "nextState " ++ show x
    stream st =
      let pw = map head st in
      if st == endState then [ pw ]
                        else pw : stream (nextState st)

hash :: String -> String
hash =
  concatMap (printf "%02x" :: Word8 -> String) .
    toOctets . sha256Ascii
