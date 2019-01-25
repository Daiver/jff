import Control.Monad.ST

import Language.Haskell.TH

import Tuple

main = do
    let f = ($(tuple 2)) 
    print ( ($(tuple 2)) $ [1, 2] )

