import Control.Monad
import Control.Arrow
import Data.Maybe

downpair :: (Monad m, Functor m) => m (a, b) -> (m a, m b)
--downpair = ((return . fst =<<) &&& (return . snd =<<)) 
downpair = fmap fst &&& fmap snd

main = do
    print $ downpair $ Just ("Hello", "World")
