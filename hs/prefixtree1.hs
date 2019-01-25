import qualified Data.List as L
import Data.Common

data Tree a = Empty | Node a Bool [Tree a] deriving (Show, Eq)

--buildTree :: [[a]] -> [Tree a]
buildTree value end dt = Node value end (map (\x -> buildTree (head . head $ x) (1 == (length . head $ x)) (map tail x) ) . grps $ dt)
    where grps = partitionBy head . filter (not . null)

treeToList (Node v is_end lst) 
                | null lst  = [[v]]
                | is_end = [v] : foldl (\l x -> (map (v:) $ treeToList x) ++ l) [] lst
                | otherwise = foldl (\l x -> (map (v:) $ treeToList x) ++ l) [] lst

searchTree tl [] tree  = Just (map (init tl ++) . treeToList $ tree)
searchTree tl (x:xs) (Node _ _ lst) 
                | null next_sub_trees = Nothing
                | otherwise = searchTree (tl ++ [x]) xs (head next_sub_trees)
    where next_sub_trees = filter (\ (Node v1 _ _) -> v1 == x) lst

main = do
    let tree = buildTree ' ' False ["hi", "hello", "me", "you", "mouse", "mek", "meee"] 
    print tree
    print $ treeToList tree 
    print $ searchTree "" "m" tree
