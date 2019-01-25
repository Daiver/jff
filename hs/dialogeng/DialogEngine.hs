module DialogEngine where

data Location state locKey= Location {
        locationText           :: state -> String,
        locationTransitionsIds :: [Int],
        locationStateChange    :: state -> state
    }

data Transition state locKey = Transition {
        transitionText         :: state -> String,
        transitionAppearing    :: state -> Bool,
        transitionStateChange  :: state -> state,
        transitionNextLocation :: state -> Int
    }

data Dialog state locKey = Dialog {
        locations   :: [Location state locKey],
        transitions :: [Transition state locKey]
    }

runLocation :: [Transition state locKey] -> Location state locKey -> state -> (state, String, [Transition state locKey])
runLocation transitions loc state = let 
        s'          = locationStateChange loc state
        text        = locationText loc s'
        transitions = filter (flip transitionAppearing s') $ map (transitions !!) $ locationTransitionsIds loc
    in (s', text, transitions)

runTransition :: Transition state locKey -> state -> (state, Int)
runTransition transition state = let 
        s' = transitionStateChange transition state
    in (s', transitionNextLocation transition s')

runDialogIO :: Dialog state locKey -> state -> Int -> IO()
runDialogIO dialog@(Dialog locs trans) initState startLocId = do
    let loc = locs !! startLocId
        (s', text, transToShow) = runLocation trans loc initState
    print text
    mapM_ (\(i, x) -> print $ show i ++ ": " ++ transitionText x s') $ zip [0..] transToShow
    transInd <- return . read =<< getLine
    let curTran = trans !! transInd
        (s'', nextLocInd) = runTransition curTran s'
    runDialogIO dialog s'' nextLocInd
