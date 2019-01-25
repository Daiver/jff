import System.Random
import qualified Data.List as List
import qualified Data.Map as Map
import Control.Monad
import Control.Arrow

import ML.Common
import ML.DecisionTree
import ML.RandomForest

main = do
    print "1"
    let samples = [(1, [1.0,2,3]), (1, [5,6,7]), (2, [0,2,3])]
        tree =  buildNode samples
    print tree
    print $ predict tree [0, 3, 4]
    
    {-rawText <- readFile "./trainset"
    let trainDataSet = reasSamplesClassIsLast $ rawText
    rawText <- readFile "./testset"
    let testDataSet = reasSamplesClassIsLast $ rawText -}
    rawText <- readFile "./wine_train"
    let trainDataSet = reasSamplesClassIsLead $ rawText
    rawText <- readFile "./wine_test"
    let testDataSet = reasSamplesClassIsLead $ rawText

    let tree = buildNode trainDataSet
    print tree
    --mapM_ (\sample -> print $ show (fst sample) ++ show (predictDecTree tree (snd sample))) testDataSet
    print $ (show $ testClassifier tree testDataSet) ++ "/" ++ (show $ length testDataSet)
