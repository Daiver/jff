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
    let gen = mkStdGen 42
    let (res, gen2) = randomSeqInRange gen 10 0 10
    let (res2, gen3) = randomSeqInRange gen2 10 0 10
    print res
    print res2
    print $ shuffle [1,2,3,4,5,6] gen3
    print $ randomSampling gen2 "abcdef"
    print $ randomSamplingWithRepeats gen2 "abcdef"
    
    --{-
    rawText <- readFile "./trainset"
    let trainDataSet = reasSamplesClassIsLast $ rawText
    rawText <- readFile "./testset"
    let testDataSet = reasSamplesClassIsLast $ rawText --}
    {-
    rawText <- readFile "./wine_train"
    let trainDataSet = reasSamplesClassIsLead $ rawText
    rawText <- readFile "./wine_test"
    let testDataSet = reasSamplesClassIsLead $ rawText --}

    --let tree = buildNode trainDataSet
    let tree = trainRandomForest (mkStdGen 2) trainDataSet
    --print tree
    --mapM_ (\sample -> print $ show (fst sample) ++ show (predictDecTree tree (snd sample))) testDataSet
    print $ (show $ testClassifier tree testDataSet) ++ "/" ++ (show $ length testDataSet)
