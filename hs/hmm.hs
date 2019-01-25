import Control.Monad
import Control.Arrow
import qualified Numeric.Matrix as N

data HMM = HMM (N.Matrix Float) (N.Matrix Float) deriving Show

viterbi :: HMM -> [Float] -> [Int] -> ([[Float]], Float, [Int])
viterbi (HMM transP emitP) initialState (initObs:tailObs) = (vLast, maximum $ last vLast, path)
    where 
        vLast = reverse $ foldl iter [v0] tailObs  
        path = map ((+1) . snd . maximum . flip zip [0 .. ]) vLast
        (countOfStates, countOfEmits) = N.dimensions emitP
        states = [1..countOfStates]

        bestNextState v0 obs y = maximum . map 
                (\y0 -> (v0 !! (y0 - 1)) * N.at transP (y0, y) * N.at emitP (y, obs)) $ states

        v0 = map (\i -> (initialState !! (i - 1)) * N.at emitP (i, initObs)) states
        iter (v0 : v) obs = vn : v0 : v
            where                   
                vn = map (bestNextState v0 obs) states

main = do
    let transP = N.fromList [
                [0.7, 0.3],
                [0.4, 0.6]
            ]
    let emitP = N.fromList [
                [0.5, 0.4, 0.1],
                [0.1, 0.3, 0.6]
            ]
    let hmm = HMM transP emitP

    print $ viterbi hmm [0.6, 0.4] [1, 2, 3]
