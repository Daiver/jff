import Control.Monad.Writer
import Debug.Trace

import Optimization

main = do
    print "Start"
    --let costF (x:_) = x**4 - 10
    let initArg = [4.4, 15]
    let costF (x:y:_) = (sin x ** 5) + (cos x) ** 3 + cos (y**2) + 2.0 + x ** 2 + abs (y ^ 2)
    --let grad (x:_) = [(x**3)*4 ]
    --print $ costF [10]
    --print $ grad [10]
    --let(res, _) = runWriter (gradientDescent (numericalGrad costF) costF 0.000000000000001 initArg)
    let(res, way) = runWriter (gradientDescent2 (numericalGrad costF) costF 0.000000000000001 initArg)
    print way
    print res
    print $ costF res
    --print $ runWriter (gradientDescent (grad) costF 0.0000001 [7.1])
