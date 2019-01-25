{- Printf.hs -}
{-# LANGUAGE TemplateHaskell #-}
module Printf where

-- Skeletal printf from the paper.
-- It needs to be in a separate module to the one where
-- you intend to use it.

-- Import some Template Haskell syntax
import Language.Haskell.TH

-- Describe a format string
data Format = D | S | L String deriving Show

-- Parse a format string.  This is left largely to you
-- as we are here interested in building our first ever
-- Template Haskell program and not in building printf.
parse :: String -> [Format]
parse [] = [L ""]
parse ('%' : 'd' : xs) = D : parse xs
parse ('%' : 's' : xs) = S : parse xs
parse (x:xs)  = L [x] : parse xs

-- Generate Haskell source code from a parsed representation
-- of the format string.  This code will be spliced into
-- the module which calls "pr", at compile time.

gen :: [Format] -> ExpQ -> ExpQ
gen        []  code = code
gen (D   : xs) code = [| \x -> $(gen xs [| $code ++ show x |]) |]
gen (S   : xs) code = [| \x -> $(gen xs [| $code ++ x |]) |]
gen (L s : xs) code = gen xs [| $code ++ s |]
 

-- Here we generate the Haskell code for the splice
-- from an input format string.
--pr :: String -> Q Exp
--pr s = gen (parse s)

pr :: String -> ExpQ
pr s = gen (parse s) [| "" |]

fact :: Integer -> ExpQ
fact 1 = [| 1 |]
fact n = [| $(fact (n - 1))*n |]


data PDBState = Str | Ident deriving Show

parseDoubleBrackets :: PDBState -> String -> String -> [(PDBState, String)]
parseDoubleBrackets Str acc ('{' : '{' : xs) = (Str, acc) : parseDoubleBrackets Ident "" xs
parseDoubleBrackets Ident acc ('}' : '}' : xs) = (Ident, acc) : parseDoubleBrackets Str "" xs
parseDoubleBrackets state acc (x : xs) = parseDoubleBrackets state (acc ++ [x]) xs
parseDoubleBrackets state acc [] = [(Str, acc)]

generateInline :: [(PDBState, String)] -> ExpQ
generateInline [] = [| "" |]
generateInline ((Str, s) : xs) = [| s ++ $(generateInline xs) |]
generateInline ((Ident, s) : xs) = [| $(dyn s) ++ $(generateInline xs) |]

printf2 :: String -> ExpQ
printf2 = generateInline . parseDoubleBrackets Str ""


printf3 :: String -> ExpQ
printf3 = generateInline . map ((,) Ident) . words

{-
{- Printf.hs -}
{-# LANGUAGE TemplateHaskell #-}
module Printf where

import Language.Haskell.TH

>>>>>>> 764bc1dcae84f4edb4fabfbd2425a34ea8613986
data PDBState = Str | Ident deriving Show
parseDoubleBrackets :: PDBState -> String -> String -> [(PDBState, String)]
parseDoubleBrackets Str acc ('{' : '{' : xs) = (Str, acc) : parseDoubleBrackets Ident "" xs
parseDoubleBrackets Ident acc ('}' : '}' : xs) = (Ident, acc) : parseDoubleBrackets Str "" xs
parseDoubleBrackets state acc (x : xs) = parseDoubleBrackets state (acc ++ [x]) xs
parseDoubleBrackets state acc [] = [(Str, acc)]

generateInline :: [(PDBState, String)] -> ExpQ
generateInline [] = [| "" |]
generateInline ((Str, s) : xs) = [| s ++ $(generateInline xs) |]
<<<<<<< HEAD
generateInline ((Ident, s) : xs) = [| $(dyn s) ++ $(generateInline xs) |]
=======
generateInline ((Ident, s) : xs) = [| show $(dyn s) ++ $(generateInline xs) |]
>>>>>>> 764bc1dcae84f4edb4fabfbd2425a34ea8613986

printf2 :: String -> ExpQ
printf2 = generateInline . parseDoubleBrackets Str ""

<<<<<<< HEAD
printf3 :: String -> ExpQ
printf3 = generateInline . map ((,) Ident) . words
=======

{-# LANGUAGE TemplateHaskell #-}

import Printf

l = [1,2,3,4]

main = do
    let s = 1
    print $(printf2 "This is {{s}}. This is {{l}}")
-}
