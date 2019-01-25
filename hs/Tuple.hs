{-# LANGUAGE TemplateHaskell #-}
module Tuple where

import Language.Haskell.TH
import Control.Monad
import Data.Maybe

--{-
tuple :: Int -> ExpQ
tuple n = [|\list -> $(tupE (exprs [|list|])) |]
  where
    exprs list = [infixE (Just (list))
                         (dyn "!!")
                         (Just (litE $ integerL (toInteger num)))
                    | num <- [0..(n - 1)]]
--- --}

{-
tuple :: Int -> ExpQ
tuple n = do
    ns <- replicateM n (newName "x")
    lamE [foldr (\x y -> conP '(:) [varP x,y]) wildP ns] (tupE $ map varE ns) --}
