import Control.Concurrent
import Control.Monad
import Data.Maybe
import System.IO
 

stuff = do
    hSetBuffering stdin NoBuffering
    c <- getChar
    print c
    stuff

main = do stuff
