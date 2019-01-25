module Plug (thing, thing2) where
import Data.Dynamic

thing :: Dynamic
thing = toDyn (1234000::Integer)

thing2 :: Dynamic
thing2 = toDyn f

f :: Int -> Int
f x = x * x
