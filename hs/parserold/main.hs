import Control.Applicative

f x y = x*x + y*y

data Tree a = EmptyNode | Node a (Tree a)  (Tree a)  deriving (Show, Read, Eq, Ord)  

instance Functor Tree where  
    fmap func (Node v l r) = Node (func v) (fmap func l) (fmap func r)
    fmap func EmptyNode = EmptyNode

treeInsert x EmptyNode = Node x EmptyNode EmptyNode
treeInsert x (Node v l r) 
    | x == v = Node v l r
    | x > v = Node v l (treeInsert x r)
    | x < v = Node v (treeInsert x l) r

fillTree  10000 tree = tree 
fillTree  x tree = let a = treeInsert x tree
                   in fillTree (x + 1) a

tree = Node 1 (Node (-2) EmptyNode EmptyNode) (Node 2 EmptyNode EmptyNode)
main = do
    --print $ [(+2), (+3)] <*> [1, 2, 3]
    print $ (+2) <$> tree
    putStr "\n"
