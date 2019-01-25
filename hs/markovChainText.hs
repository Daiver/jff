import Control.Monad
import Control.Arrow
import Control.Applicative
import Control.Monad
import qualified Data.Map as Map

import System.Random
import System.Environment

counts li = Map.fromListWith (+) $ [(c, 1) | c <- li]

symbols = ".,/\\?|=+-_)(*&^%$#@!'\";:><"
spaces = " \t\n"

data ParserState = PSSpaces | PSSymbols String | PSLetters String deriving (Show, Eq)

unpackState = map f . filter (/= PSSpaces)
    where
        f (PSSymbols s) = s
        f (PSLetters s) = s

parseString :: ParserState -> String -> [ParserState]
parseString PSSpaces (x : xs) 
    | x `elem` spaces  = parseString PSSpaces xs
    | x `elem` symbols = parseString (PSSymbols [x]) xs
    | otherwise        = parseString (PSLetters [x]) xs
parseString (PSSymbols s) (x : xs)
    | x `elem` spaces  = (PSSymbols $ reverse s) : parseString PSSpaces xs
    | x `elem` symbols = parseString (PSSymbols (x : s)) xs
    | otherwise        = (PSSymbols $ reverse s) : parseString (PSLetters [x]) xs
parseString (PSLetters s) (x : xs)
    | x `elem` spaces  = (PSLetters $ reverse s) : parseString PSSpaces xs
    | x `elem` symbols = (PSLetters $ reverse s) : parseString (PSSymbols [x]) xs
    | otherwise = parseString (PSLetters (x : s)) xs
parseString PSSpaces [] = []
parseString st [] = [st]

splitWords = unpackState . parseString PSSpaces

makePairs :: [a] -> [(a, a)]
makePairs (x : y : xs) = (x, y) : makePairs (y : xs)
makePairs (x : []) = []
makePairs [] = []

makeChains :: (Ord a) => [a] -> Map.Map (a, a) [a]
makeChains  = Map.fromListWith (++) . f
    where 
        f (x : y : z : xs) = ((x, y), [z]) : f (y : z : xs)
        f (x : y : []) = []

genRandText :: StdGen -> Map.Map (String, String) [String] -> Int -> String
genRandText g dict size = unwords $ [fst initPair, snd initPair] ++ f g' initPair size
    where
        keys = Map.keys dict
        (r1, g') = random g
        initPair = keys !! (r1 `mod` length keys)
        runMaybe Nothing x  = x
        runMaybe (Just x) _ = x
        f _ _ 0 = []
        f gen pair@(a, b) n = next : f gen' (b, next) (n - 1)
            where 
                (r, gen') = random gen
                li = runMaybe (Map.lookup pair dict) [a]
                next = li !! (r `mod` length li)

test = do
    print $ splitWords "123. 321     uiou\nlklklklk(me and you)"
    print $ makePairs [1,2,3, 4, 5]
    print $ makeChains $ splitWords "123. 321     uiou\nlklklklk(me and you)"

main = do
    --test
    s <- readFile . head =<< getArgs
    let dict = makeChains . splitWords $ s
    print $ genRandText (mkStdGen 42) dict 100
