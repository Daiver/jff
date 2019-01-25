module Lexer (tokenize, symbols) where

import qualified Data.Set as Set
import qualified Data.Char as Char

data TokenizerState = Separator | LatinDigit | Symbol deriving (Eq, Show)

symbols    = Set.fromList "!@#$%^&*()_+=-\"';:.,></?\\|~`"
separators = Set.fromList " \n\t"

tokenizeInner :: (TokenizerState, String, [String]) -> Char -> 
                            (TokenizerState, String, [String])
tokenizeInner (state, buf, lst) c 
    | curr == Separator && state == Separator                = (Separator, "", lst)
    | curr == Separator && state `elem` [LatinDigit, Symbol] = (Separator, "", buf:lst)
    | curr == state                                          = (curr, c:buf, lst)
    | curr /= state && state == Separator                    = (curr, [c], lst)
    | curr /= state                                          = (curr, [c], buf:lst)
    where curr 
            | Set.member c separators           = Separator
            | Set.member c symbols              = Symbol
            | Char.isDigit c || Char.isLetter c = LatinDigit

tokenize :: String -> [String]
tokenize s = reverse . map reverse $ res
    where 
        (_, b, l) = foldl tokenizeInner (Separator, "", []) s
        res = if not $ null b then b:l else l

