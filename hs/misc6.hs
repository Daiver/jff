import Control.Applicative
import Data.Maybe

data Tree a = Empty | Node a (Tree a) (Tree a) deriving Show
instance Functor Tree where
    fmap f (Node v l r) = Node (f v) (fmap f l) (fmap f r)
    fmap f Empty = Empty

insertToTree Empty item = Node item Empty Empty
insertToTree (Node v l r) item 
    | v < item = Node v l (insertToTree r item)
    | otherwise = Node v (insertToTree l item) r

insertSubTree t Empty = t
insertSubTree (Node v l r) t = insertSubTree r $ insertSubTree l $ insertToTree t v

instance Monad Tree where
    return x = Node x Empty Empty
    (>>=) (Node v l r) f = f v
    (>>) _ _ = Empty
    fail _ = Empty

data ErrorHandler a = NotError a | Error String deriving Show

instance Monad ErrorHandler where
    return x = NotError x
    (>>=) m f = case m of
                    NotError x -> f x
                    Error m  -> Error m 
    fail m = Error m

foo = (/)

safeFoo _ 0 = Nothing
safeFoo x y = Just (foo x y)

safeAvg seq = safeFoo (sum seq) (fromIntegral . length $ seq)

main = do
    print ""
    --print $ (NotError 10) >> (Error "oops")
    --let tree = Node 10 (Node 1 Empty Empty) (Node 90 Empty Empty)
    --print $ fmap (*2) tree
    {-let tree = Node 10 (Node 1 Empty Empty) (Node 90 Empty Empty)
    print $ fmap (*2) tree
    print $ insertToTree tree 9
    print $ tree >>= (\v -> return (v * 2))
    print $ tree >> tree
    print "End"
    -}
