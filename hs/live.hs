import Control.Monad
import Control.Monad.IO.Class (liftIO)
import System.Environment

type Field = [[Int]]
type Pos = (Int, Int)

countNeighbourhs :: Field -> Pos -> Int
countNeighbourhs field (x, y) = sum . map (\ (x, y) -> field !! y !! x) $ new_pos 
    where
        steps = [(1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]
        new_pos  = filter (\(x, y) -> x >= 0 && y >= 0 && x < length (head field) && y < length field) $ map (\ (nx, ny) -> (nx + x, ny + y)) steps

nextGeneration :: Field -> Field
nextGeneration field = map (\ i -> map (\j -> rule (field !! i !! j) $ countNeighbourhs field (j, i)) [0..(length (head field) - 1)]) [0..length field - 1]
    where
        rule value count
            | count == 3 && value == 0 = 1
            | value == 1 && count `elem` [2, 3] = 1
            | otherwise = 0

readFieldFromString :: String -> Field
readFieldFromString = map (map read . words) . lines 

printField :: Field -> IO [()]
printField = mapM (print . map (\x -> if x == 1 then '1' else ' '))

ioCycle :: Field -> IO Field
ioCycle field = do
    getLine
    let f = nextGeneration field
    print $ replicate (length . head $ field) '-'
    printField f
    print $ replicate (length . head $ field) '-'
    ioCycle f

main = do
    content <- readFile . head =<< getArgs
    print "Start"
    let field  = readFieldFromString content
    printField field
    ioCycle field
