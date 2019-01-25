import Data.Ord
import Data.List
import qualified Data.Map as M
import Debug.Trace

freqDict input = M.fromListWith (+) [(c, 1) | c <- input]

data DecisionTree = Node Int Float DecisionTree DecisionTree | Leaf [(Int, Int)]
    deriving Show

activate :: DecisionTree -> [Float] -> [(Int, Int)]
activate (Node col val l r) sample 
    | (sample !! col) > val = activate r sample
    | otherwise             = activate l sample

activate (Leaf res) _ = res

buildNode :: [([Float], Int)] -> DecisionTree
buildNode train_data 
    | gain > 0  = 
        let (a, b) = partition (\x -> (fst x) !! col < val) train_data 
        in Node col val (buildNode a) (buildNode b)
    | otherwise = Leaf $ M.toList freqs
    where
        data_list   = map fst train_data
        labels_list = map snd train_data
        sz          = fromIntegral $ length  train_data
        f_sz        = length $ head data_list
        gini :: M.Map Int Int -> Float
        gini mp = (1.0 - (sum . map (\x -> ((fromIntegral x / sz)) ** 2) $ M.elems mp))
        sortByFeature fi = init $ sortBy (ficompare fi) [0..length data_list - 1]
        ficompare fi x y = 
            if (data_list !! x !! fi) > (data_list !! y !! fi) then GT else LT
        freqs = freqDict labels_list
        vInit = gini freqs
        start_params = (0, freqs, M.fromList . map (flip (,) 0) $ M.keys freqs)
        start_state = ( 
                        start_params, 0, 0, vInit -- col and val and gain
                      ) 
        findThrFInner ((wls, wr, wl), col, val, gain) (fi, idx) 
        --findThrFInner (fi, idx) ((wls, wr, wl), col, val, gain) 
            | wls < sz = ((wls + 1, wr', wl'), col', val', gain')
            | otherwise = (start_params, col', val', gain')
            where
                label = trace ("lb" ++ show (labels_list !! idx) ++ " " ++ show wls) $ labels_list !! idx
                changeW cnst = M.update (return . (+cnst)) label
                wr' = trace ("wr " ++ (show . M.toList $ wr)) changeW (negate 1) wr
                wl' = changeW 1 wl
                v = gini wl' * wls / sz + gini wr' * (sz - wls) / sz
                (col', val', gain') = if v < gain 
                    then (fi, data_list !! idx !! fi, v) 
                    else (col, val, gain)
        idxs = [(f, i) | f <- [0 .. f_sz - 1], i <- sortByFeature f]
        (_,col, val, bgain) = foldl findThrFInner start_state $ trace ("idxs " ++ show idxs) idxs
        --gain = bgain
        gain = trace ("vInit " ++ show vInit) vInit - trace ("bgain " ++ show bgain) bgain

main = do
    print "Start"
    let train_data = [
            ([3,0], 1)
            , ([2,1], 1)
            , ([0,0.9], 0)]
    let tree = buildNode train_data
    print tree
    print $ activate tree [2,2]
    print $ activate tree [0,1]
    print "End"
