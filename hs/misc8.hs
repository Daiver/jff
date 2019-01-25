import Control.Monad
{-import Control.Functor-}

main  = do
    print "Hi"
    val <- read <$> getLine 
    {-val <- fmap read $ getLine -}
    {-val <- getLine >>= (return . read)-}
    print $ val + 1
