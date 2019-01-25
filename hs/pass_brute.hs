import Control.Monad
import Control.Parallel
import Control.Parallel.Strategies
import Bruteforce.Common( hash )

type Hash = String
type Password = String

filterMatched :: [Hash]
              -> [Password]
              -> [(Password, Hash)]
filterMatched knownHashes candidates = 
        filter (elemOf knownHashes . snd) pwHashList
        `using` parList rdeepseq
    where 
        hashes = map hash candidates
        pwHashList = zip candidates hashes
        elemOf = flip elem

main :: IO ()
main = do
    let hashList = [
            -- 1234
            "03ac674216f3e15c761ee1a5e255f067" ++
            "953623c8b388b4459e13f978d7c846f4",
            -- r2d2
            "8adce0a3431e8b11ef69e7f7765021d3" ++
            "ee0b70fff58e0480cadb4c468d78105f"]
        pwLen = 4
        charList = ['0'..'9'] ++ ['a'..'z']
        allPasswords = replicateM pwLen charList
        matched = filterMatched hashList allPasswords

    mapM_ (putStrLn . showMatch) matched

    where showMatch (pw, h) = pw ++ ":" ++ h

