import System.Environment

import Data.Common

data GMap = GMap [[ Int ]] deriving Show

showGMap (GMap value) = do sequence . map print . map procLine $ value
    where 
        charFromVal 0 = " _ "
        charFromVal 1 = " X "
        charFromVal 2 = " O "
        procLine l = (foldl1 (++) . map charFromVal $ l) ++ "\n"

zeroGMap n = GMap ( replicate n . replicate n $ 0)

features = map show [1..9]

main = do
    showGMap $ zeroGMap 3
