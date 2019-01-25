module Lexer (tokenize, removeUselessTokens) where 

import Data.Char

import Common

--import Tuple

{-
type Token = (Int, Int, String)

symbols = "\"'-=+_;:!@$%^&*/?.,<>"
brackets = "(){}[]"
spaces = "\t "
newLine = "\n"
-}

tokenize :: String -> [Token]
tokenize s = allTokens
    where 
        allTokens = reverse (token : init tokens)
        (tokens, token, _, _, _) = foldl tokenizeInner ([], (1,1,""), LNewLine, 1, 0) s

data LexerStat = LLetter | LNumber | LSymbol | LSpace | LNewLine | LBrackets
    deriving Eq

tokenizeInner :: ([Token], Token, LexerStat, Int, Int) ->
                 Char -> ([Token], Token, LexerStat, Int, Int)

tokenizeInner (res,t,lstate,countOfLines,countOfChar) c
    | isLetter c && lstate /= LLetter =
              (t : res, (countOfLines , countOfChar + 1, [c]), LLetter, countOfLines, countOfChar + 1)
    | isDigit c && lstate /= LNumber =
              (t : res, (countOfLines , countOfChar + 1, [c]), LNumber, countOfLines, countOfChar + 1)
    | c `elem` symbols && lstate /= LSymbol =
              (t : res, (countOfLines , countOfChar + 1, [c]), LSymbol, countOfLines, countOfChar + 1)
    | c `elem` spaces && lstate /= LSpace =
              (t : res, (countOfLines , countOfChar + 1, [c]), LSpace, countOfLines, countOfChar + 1)
    | c `elem` brackets =
              (t : res, (countOfLines , countOfChar + 1, [c]), LBrackets, countOfLines, countOfChar + 1)
    | c `elem` newLine =
              (t : res, (countOfLines , countOfChar + 1, [c]), LNewLine, countOfLines + 1, 0)
    | otherwise = let (line, count, str) = t in 
              (res, (line , count, str ++ [c]), lstate, countOfLines, countOfChar + 1)



removeUselessTokens, removeUselessTokensInner, removeComments :: [Token] -> [Token]

removeUselessTokens = thrd . snd . fst
    where 
        fst  = removeUselessTokensInner 
        snd  = dropWhile 
            (\(_,_,t) -> (t == "\n" || all (`elem` spaces) t ))
        thrd = reverse . snd . reverse

removeComments = id

removeUselessTokensInner [] = []
removeUselessTokensInner [x] = [x]
removeUselessTokensInner (x@(_, _, "\n") : y@(_, _, "\n") : xs) = removeUselessTokensInner (y : xs)

removeUselessTokensInner (x@(_, _, xt) : y@(_, _, yt) : xs) 
    | all (`elem` spaces) yt && xt == "\n" = x : y : removeUselessTokensInner xs
    | all (`elem` spaces) xt = removeUselessTokensInner (y:xs)
    | otherwise = x : removeUselessTokensInner (y:xs)

