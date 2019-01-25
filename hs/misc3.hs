import Control.Applicative
import Control.Monad.State
import Data.Maybe

import System.Environment       

data Stack a = Stack [a] deriving Show

push :: a -> State (Stack a) ()
push x = get >>= (\(Stack v) -> put (Stack (x:v)))

pop :: State (Stack a) a
pop = do
    (Stack val) <- get
    put (Stack (tail val))
    return (head val)

emptyStack :: State (Stack a) Bool
emptyStack = do
    (Stack li) <- get
    put (Stack li)
    return (null li)

data PSActions = PSPlus | PSMinus | PSMul | PSDiv deriving Show
data PolSeq = PSVal Int | PSBracket Char | PSAct PSActions deriving Show

calcPS :: PolSeq -> State (Stack Int) ()
calcPS h = case h of
        (PSVal v) -> push v
        (PSAct act) -> do
            a <- pop
            b <- pop
            push $ case act of
                PSPlus  -> a + b
                PSMinus -> a - b
                PSMul   -> a * b
                PSDiv   -> a `div` b

calcPSs :: [PolSeq] -> State (Stack Int) Int
calcPSs [] = pop
calcPSs (x:xs) = calcPS x >> calcPSs xs

readOnePS "+" = PSAct PSPlus
readOnePS "-" = PSAct PSMinus
readOnePS "*" = PSAct PSMul
readOnePS "/" = PSAct PSDiv
readOnePS "(" = PSBracket '('
readOnePS ")" = PSBracket ')'
readOnePS x = PSVal (read x)

readPS :: String -> [PolSeq]
readPS = map readOnePS . words

lexIt symbols spaces input = filter (\x -> not (null x || (head x) `elem` spaces)) $ lexItInner [] "" input
    where 
        splitters = spaces ++ symbols
        lexItInner res cur [] = res ++ [cur]
        lexItInner res cur (x:xs) 
            | x `elem` splitters = lexItInner (res ++ [cur, [x]]) "" xs
            | otherwise = lexItInner res (cur ++ [x]) xs


testF :: State (Stack PolSeq) [PolSeq] 
testF = do
    sym <- pop
    emp <- emptyStack
    if emp then return []
    else
        case sym of 
            (PSVal v) -> testF >>= \x -> return ((PSVal v) : x)  
            (PSAct a) -> return []

{-parsIt l = expression l
    where
        expression (x:xs) = 
-}

main = do
    let symbols = "()+=-/*"
    let spaces  = " \n"
    print $ runState testF (Stack [PSVal 10])
    print $ map readOnePS $ lexIt symbols spaces "1 + 2 * (5-2)/100"
    --seq <- readPS . head <$> getArgs --"1 2 3 + 2 * +"
    --print $ runState (calcPSs seq) (Stack [])
