import qualified Data.Map as Map
import Data.Maybe

type Func = ([Float] -> [Float])

runPolish :: Map.Map String Func -> [String] -> [Float] -> ([String], [Float])
runPolish _ [] stack = ([], stack)
runPolish funcs (curToken : otherTokens) stack = case command of 
    Nothing -> runPolish funcs otherTokens (read curToken : stack)
    Just f  -> runPolish funcs otherTokens $ f stack
    where 
        command = Map.lookup curToken funcs
        

main = do
    let funcs = Map.fromList [
                ("+", \(x:y:xs) -> (x + y) : xs),
                ("-", \(x:y:xs) -> (x - y) : xs),
                ("*", \(x:y:xs) -> (x * y) : xs),
                ("/", \(x:y:xs) -> (x / y) : xs)
            ]
    let seq = "6 1 2 3 + - * 2.0 /"
    print $ snd . runPolish funcs (words seq) $ []
    print "123"
