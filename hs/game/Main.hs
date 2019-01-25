module Main where

import GameTypes
import Ship
import Common

routineCycle (GameWorld system)= do
    print "HI"
    putStrLn $ drawSystemMap system

main = do
    let sys       = SystemMap [SpaceObject "Tatuin" (5, 5) 1 '*']
    let initState = GameWorld sys
    routineCycle initState
