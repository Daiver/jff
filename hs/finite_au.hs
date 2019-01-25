{-# OPTIONS_GHC -XMultiParamTypeClasses #-}

class FiniteAutomata a b where
    (<=<) :: a -> b -> a



main = do
    print "Hi"
