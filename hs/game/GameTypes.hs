{-# LANGUAGE TemplateHaskell #-}

module GameTypes where

import qualified Data.Map as Map
import qualified Data.List as List
import Control.Monad
import Control.Lens
import Control.Arrow

import Common

type Point = (Int, Int)

data SpaceSize = SatelliteLike | VerySmall | Small | Medium | Large | VeryLarge

data SpaceObject = SpaceObject {
    _name     :: String,
    _position :: Point,
    _radius   :: Int,
    _symbol   :: Char
} deriving Show
makeLenses ''SpaceObject

data SOPlanetoid = SOPlanetoid {
    _so :: SpaceObject
}
makeLenses ''SOPlanetoid

data SystemMap = SystemMap {
    _space :: [SpaceObject]
} deriving Show
makeLenses ''SystemMap

coordMapFromSpaceObjects :: [SpaceObject] -> Map.Map Point SpaceObject
coordMapFromSpaceObjects = Map.fromList . map (_position &&& id)

nameMapFromSpaceObjects :: [SpaceObject] -> Map.Map String SpaceObject
nameMapFromSpaceObjects = Map.fromList . map (_name &&& id)

drawSystemMap :: SystemMap -> String
drawSystemMap (SystemMap space) = unlines res
    where
        coordMap = coordMapFromSpaceObjects space
        keys        = Map.keys $ coordMap
        h           = List.maximum . map fst $ keys
        w           = List.maximum . map snd $ keys
        res         = [[getSymb i j | j <- [0..w]] | i <- [0..h]]
        getSymb i j = case Map.lookup (i, j) coordMap of
            Nothing -> ' '
            Just x  -> _symbol x

data GameWorld = GameWorld {
    _currentSystem :: SystemMap 
}
makeLenses ''GameWorld

{-
position :: Lens' Actor Point
position = lens _position (\actor p -> if predicat p then actor { _position = p } else actor )
-}
