module Data.AStar where

import Control.Applicative
import Control.Monad
import qualified Data.List as L
import qualified Data.Set as S
import qualified Data.List.Ordered as O
import Data.Maybe
import Data.Ord

data AStarNode a = AStarNode {value :: a, distance :: Int, path :: [a]} deriving (Show, Eq, Ord)

aStar isFinish g h expand init = (value <$> fst res, path <$> fst res, snd res)
    where 
        res = aStarInner S.empty [AStarNode init 0 []]
        aStarInner closed opened 
            | null opened = (Nothing, opened)
            -- | 9 < distance node = (Nothing, opened)
            -- | length new_opened == length opened = (Just node, new_opened)
            | isFinish $ value node = (Just node, opened)
            | otherwise = aStarInner (S.insert (value node) closed) new_opened
            where
                node = selectBest opened
                new_opened = L.nubBy (\(AStarNode x _ _) (AStarNode y _ _) -> x == y) $ foldl (insertToOpened closed) (O.minus opened [node]) (map (\x ->  AStarNode x (distance node + g x (value node)) (value node : path node) ) (expand $ value node))
                insertToOpened closed opened item
                    | S.member (value item) closed = opened 
                    | otherwise = O.insertSetBy (\ (AStarNode x dx _) (AStarNode y dy _) -> if (dx + h x) > (dy + h y) then GT else LT) item opened
                selectBest = last --L.maximumBy (\ (AStarNode x dx _) (AStarNode y dy _) -> if (dx + h x) > (dy + h y) then GT else LT)

