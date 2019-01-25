module Main where
import Test.HUnit

import qualified Data.Map as Map

import Common
import GameTypes

commonTest1 = TestCase (assertEqual ""
            (updateList 15 0 [1,2,3,4]) [15, 2, 3, 4])

commonTest2 = TestCase (assertEqual ""
            (updateList 15 3 [1,2,3,4]) [1, 2, 3, 15])

commonTest3 = TestCase (assertEqual ""
            (updateList2D 15 (1, 2) [[1,2,3], [4,5,6]]) 
            [[1,2,3],[4,5,15]])

commonTest4 = TestCase (assertEqual ""
            (updateList2D 15 (1, 0) [[1,2,3], [4,5,6]]) 
            [[1,2,3],[15,5,6]])

commonTest5 = TestCase (assertEqual ""
            (updateList2D 15 (0, 1) [[1,2,3], [4,5,6]]) 
            [[1,15,3],[4,5,6]])

gameTypesTest1 = TestCase (assertEqual "" (drawSystemMap mp) res)
    where
        res = "   \nX M\n"
        mp = SystemMap $ [
                            SpaceObject "H" (1,2) 1 'M',
                            SpaceObject "Ho" (1,0) 1 'X'
                        ]

tests = TestList [
              TestLabel "commonTest1" commonTest1
            , TestLabel "commonTest2" commonTest2
            , TestLabel "commonTest3" commonTest3
            , TestLabel "commonTest4" commonTest4
            , TestLabel "commonTest5" commonTest5
            , TestLabel "gameTypesTest1" gameTypesTest1
        ]

main = runTestTT tests

