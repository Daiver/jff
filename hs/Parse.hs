{- Parse.hs -}
{-# LANGUAGE TemplateHaskell #-}
module Parse where
import Data.Bool
import Data.List
import Control.Arrow
import Control.Applicative
import Data.Monoid
import Language.Haskell.TH
import Language.Haskell.TH.Syntax

bool :: a -> a -> Bool -> a
bool a b p
    | not p = a
    | otherwise = b

data Node = Str String 
          | Ident String 
          deriving Show

instance Lift Node where
    lift (Str s) = [|s|]
    lift (Ident i) = [|show $(dyn i)|]

parse :: String-> [Node]
parse str = parse' ((Str,"{{"),(Ident,"}}")) $ group str
    where parse' (s@(ws, bs), e@(we,be)) = 
              break (==bs) >>> 
              (ws . concat) *** (bool <$> (parse' (e,s) . tail) <*> const [] <*> (==[])) >>^ uncurry (:)

generate :: [Node] -> ExpQ
generate nodes = [|mconcat nodes|]
