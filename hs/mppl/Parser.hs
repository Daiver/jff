module Parser where

import Data.Maybe
import qualified Data.Map as Map
import qualified Data.Char as Char
import qualified Data.Set as Set

import Definitions
import BuildInFunctions
import Lexer

functionsNames = Map.fromList . map (\x -> (getFunctionName x, x)) $ buildInFunctions

isNumber = all Char.isDigit
isIdent (x:xs) = Char.isLetter x && all (\c -> Char.isLetter c || Char.isDigit c) xs
isOperator = all (flip Set.member symbols)

parse :: [String] -> Operation Float
parse stream = Constant 0
    where
        innerParse (stx:stxs@stack, out) token = 1
