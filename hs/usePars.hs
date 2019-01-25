{-# LANGUAGE TemplateHaskell #-}

import Parse

everybody = 1
someList = [3,2,1]
main = print $(generate $ parse "hello here {{everybody}} !111 {{someList}} .")
