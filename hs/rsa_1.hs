import Data.List
import System.Random
import Data.FixedPoint
import GHC.Word
import Math.NumberTheory.Primes.Counting
import Math.NumberTheory.Primes.Testing
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BSC

packBigWord :: Integer -> (Word64, Word64) -> Integer
packBigWord r (x, i) = r + (fromIntegral x) * 2^(i*64) 

prep :: [Word64] -> [Integer]
prep l = res:(prep $ drop 8 l)
    where 
        res = foldl packBigWord  0 $ 
            zip (take 8 l) (reverse [0..8] )

prep2 :: [Int] -> [Int]
prep2 (a:r) = (abs a):(prep2 r)

align a b x = a + (x `mod` b)

findTandSFromN n = 
    let 
        tmp x s
            | x `mod` 2 == 0 = tmp (x `div` 2) (s + 1)
            | otherwise = (x, s)
    in tmp (n-1) 0

invm :: Integer -> Integer -> Integer
invm m a
    | g /= 1 = error "No inverse exists"
    | otherwise = x `mod` m
    where (g,x,_) = gcde a m


expm :: Integer -> Integer -> Integer -> Integer
expm m b k =
    let
        ex a k s
            | k == 0 = s
            | k `mod` 2 == 0 = ((ex (a*a `mod` m)) (k `div` 2)) s
            | otherwise = ((ex (a*a `mod` m)) (k `div` 2)) (s*a `mod` m)
    in ex b k 1


equalTByMod a t b n = ((a ^ t) `mod` n) == b
test5 a n s t = not . null $ filter (\k -> (expm n a ((2^k * t))) ==  (negate 1) ) [0..s]
testA a n s t = not ( a < 1 || (expm n a t) == 1 || (test5 a n s t))

--specTest :: [Int] -> [(Bool, (Int, Int))]
specTest :: [Integer] -> [(Bool, (Integer, Integer))]
specTest (n:a:xs) -- = (1,1)
        | n' `mod` a' == 0  = (False, (n', a')) : specTest xs
        -- | isFermatPP n' a'  = [(True, (n', a'))]
        | isStrongFermatPP n' a' = [(True, (n', a'))]
        -- | isFermatPP n' a' = [(True, (n', a'))]
        -- | testA a' n' s t = [(True, (n', a'))]
        | otherwise       = (False, (n', a')) : specTest (n:xs)
    where
        n' = if n `mod` 2 == 0 then n + 1 else n
        a' = n' `mod` a--align 0 (n' `div` 2) a
        (t, s) = findTandSFromN n'

genRSAKey :: Integer -> Integer -> (RSAPrivateKey,RSAPublicKey)
genRSAKey p q =
    let
        phi = (p-1)*(q-1)
        n = p*q
        e = find (phi `div` 5)
        d = invm phi e
        find x
            | g == 1 = x
            | otherwise = find ((x+1) `mod` phi)
            where (g,_,_) = gcde x phi
    in
        (PRIV n d,PUB n e)

gcde :: Integer -> Integer -> (Integer,Integer,Integer)
gcde a b =
    let
        gcd_f (r1,x1,y1) (r2,x2,y2)
            | r2 == 0 = (r1,x1,y1)
            | otherwise =
            let
                q = r1 `div` r2
                r = r1 `mod` r2
            in
                gcd_f (r2,x2,y2) (r,x1 - q * x2, y1 - q*y2)
        (d,x,y) = gcd_f (a,1,0) (b,0,1)
    in
        if d < 0
        then (-d, -x, -y)
        else (d,x,y)

data RSAPublicKey = PUB Integer Integer deriving Show
data RSAPrivateKey = PRIV Integer Integer deriving Show

ersa :: RSAPublicKey -> Integer -> Integer
ersa (PUB n e) x =  expm n x e

drsa :: RSAPrivateKey -> Integer -> Integer
drsa (PRIV n d) x = expm n x d

main = do
    g <- newStdGen 
    g1 <- newStdGen 
    --print $ head . take 1 . prep $ randoms g --(0, 2^64) g 
    print $ take 5 . prep $ randoms g --(0, 2^64) g 
    --print $ nthPrimeApprox . head . take 1 . prep $ randoms g --(0, 2^64) g 
    let p = fst . snd . head . filter fst . specTest . prep $ randoms g 
    let q = fst . snd . head . filter fst . specTest . prep $ randoms g1
    print $ expm 7 2 5
    
    let (priv, pub) = genRSAKey p q
    print (priv, pub)
    let enc = ersa pub 100
    let dnc = drsa priv enc
    print enc
    print dnc
