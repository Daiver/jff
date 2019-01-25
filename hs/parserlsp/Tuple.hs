{-# LANGUAGE TemplateHaskell #-}

module Tuple where

import Language.Haskell.TH
import Control.Monad
import Data.Maybe

sel i n = lamE [pat] rhs
    where pat = tupP (map varP as)
          rhs = varE (as !! (i - 1))
          as  = [ "a" ++ show j | j <- [1..n] ]
