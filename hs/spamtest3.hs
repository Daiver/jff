import qualified Data.Map as M
import Control.Monad
import Data.Common
import System.Environment


train samples = (normFreqDict samples classes, M.mapWithKey (\(k, w) v -> v / ( justOrValue 0 $ M.lookup k classes)) freq)
    where 
        classes = freqDict . map fst $ samples
        freq = freqDict . join . map (\(label, feats) -> map (\feat -> (label, feat)) feats) $ samples

activate classes freq feats = (ans, val)
    where
        testOne klass val = (negate . log $ val) + (sum . map (\feat -> negate . log . justOrValue 0.0000001 $ M.lookup (klass, feat) freq ) $ feats)
        (val, ans) = minimum . map (\(x, y) -> (y, x)) . M.toList . M.mapWithKey testOne $ classes

readData = convert . raw_data
    where 
        raw_data = map words . lines
        convert = map (\x -> (head x, tail x))

actAll samples classes freq = map (\(ans, dt) -> (ans, activate classes freq dt)) samples
compareAll res = foldl tmp 0 res 
    where 
        tmp err (right, (ans, val)) 
            | right == ans = err
            | otherwise = err + 1

main = do
    dt <- readFile . head =<< getArgs
    let samples = readData dt
    print "Num of samples"
    print $ length samples
    let (classes, freq) = train $ take 4000 samples
    print $ M.keys classes
    print $ length . M.keys $ freq
    let resTst = actAll (drop 4000 samples) classes freq
    --mapM print resTst
    print $ compareAll resTst
