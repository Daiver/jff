
import Control.Arrow
import System.Environment

data CPPWithDepends = CPPWithDepends {
        fname :: String,
        fpath :: String,
        dependencies :: [CPPWithDepends]
    } deriving Show

includePattern = "#include"

isIncludeStr s = and $ map ($ s) [
            (==2). length . filter (=='"'), 
            (==includePattern) . take (length includePattern)]

parseIncludePattern = takeWhile (/='"') . stripped where
    stripped = drop (length includePattern + 2)

getSubTreeFromFile fname = do
    includes <- readFile fname 
             >>= return . map parseIncludePattern . filter isIncludeStr . lines 
                
    return includes

main = do
    print "Start"
