module ParserLsp where

import Control.Monad
import Data.Maybe

import Common


data ATTBracketsType   = ATTBSimple | ATTBSquare | ATTBComplex | ATTBSpaces
data AbstractTokenTree = ATTNode ATTBracketsType Token [Token]

buildLispLikeATT, buildLispLikeATTInner :: [Token] -> AbstractTokenTree
buildLispLikeATT = undefined

buildLispLikeATTInner = undefined
