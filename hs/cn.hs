import Data.Either
import Data.Maybe

data Field = Field [[Char]]
data Point = Point Int Int

printField :: Field -> IO ()
printField (Field f) = print "===" >> mapM print f >> print "==="

ffield = Field . replicate 3 . replicate 3 $ ' '

addToken (Field f) (Point x y) token = 
    let 
        checkAndReplace i j 
            | i == x && j == y = token
            | otherwise = f !! i !! j
    in Field $ map (\i -> map (checkAndReplace i) [0..length (f !! 0) - 1]) [0..length f - 1]

changePlayer 'O' = 'X'
changePlayer 'X' = 'O'

pointFromKeyboard :: IO (Point)
pointFromKeyboard  = print "Type [i j]" >> getLine >>= processStr
    where processStr str = let [x : y : _] = map read . words $ str in return $ Point x y



main = do
    print "Start"
    printField ffield
    let f2 = addToken ffield (Point 1 1) 'X'
    printField f2
    let f3 = addToken f2 (Point 0 1) 'O'
    printField f3
    let f4 = addToken f3 (Point 2 0) 'X'
    printField f4
