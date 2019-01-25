{-# LANGUAGE TemplateHaskell #-}

import Printf

l = [1,2,3,4]

hello = "Fuzz"
world = "Buzz"

main = do
    print $ parseDoubleBrackets Str "" "12345 {{To}} 54321" 
    let s = 1
    --print $(printf2 "This is {{s}}. This is {{l}}")
    print $(printf3 "hello world")
