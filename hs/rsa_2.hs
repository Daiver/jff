data RSAPublicKey = PUB Integer Integer -- (n,e)
--data RSAPrivateKey = PRIV Integer Integer -- (n,d)
data RSAPrivateKey = PRIV Integer Integer -- (n,d)
    | CRT Integer Integer Integer Integer Integer Integer Integer
    -- (n,d,p,q,d mod (p−1),d mod (q−1),(inverse q) mod p)


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

invm :: Integer -> Integer -> Integer
invm m a
    | g /= 1 = error "No inverse exists"
    | otherwise = x `mod` m
    where (g,x,_) = gcde a m

--expm :: Integer -> Integer -> Integer -> Integer
--expm m b k = (b ^ k) `mod` m


expm :: Integer -> Integer -> Integer -> Integer
expm m b k =
    let
        ex a k s
            | k == 0 = s
            | k `mod` 2 == 0 = ((ex (a*a `mod` m)) (k `div` 2)) s
            | otherwise = ((ex (a*a `mod` m)) (k `div` 2)) (s*a `mod` m)
    in ex b k 1

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

computeRSAKey :: Integer -> Integer -> Integer -> (RSAPrivateKey,RSAPublicKey)
computeRSAKey p q e =
    let
        phi = (p-1)*(q-1)
        (g,_,_) = gcde e phi
        n = p*q
        d = invm phi e
    in
        if (g /= 1)
        then error "Public exponent not acceptable"
        else (PRIV n d,PUB n e)

ersa :: RSAPublicKey -> Integer -> Integer
ersa (PUB n e) x = expm n x e

drsa :: RSAPrivateKey -> Integer -> Integer
drsa (PRIV n d) x = expm n x d

drsa (CRT n d p q exp1 exp2 coeff) x =
    let
        (a1,a2) = (expm p x exp1,expm q x exp2)
        u = ((a2-a1)*coeff) `mod` q
    in
        a1 + u*p


class Split a where
    split :: Integer -> a -> [Integer]
    combine :: Integer -> [Integer] -> a

class PRNG g where
    nextB :: g -> (Integer,g)
    nextI :: Int -> g -> (Integer,g)
    nextM :: Integer -> g -> (Integer,g)

e_rsa :: (Split a) => RSAPublicKey -> a -> [Integer]
e_rsa k@(PUB n _) x = map (ersa k) (split n x)

d_rsa :: (Split a) => RSAPrivateKey -> [Integer] -> a
d_rsa k@(PRIV n _) x = combine n (map (drsa k) x)

data BBSRand = BBS Integer Integer -- (modulus,x)

seedBBSRand :: Integer -> Integer -> BBSRand
seedBBSRand modulus seed =
    let
        (g,_,_) = gcde seed modulus
    in
        if g /= 1
        then seedBBSRand modulus (seed + 1)
        else BBS modulus ((seed*seed) `mod` modulus)

nextBBSBit :: BBSRand -> (Integer,BBSRand)
nextBBSBit (BBS modulus x) = (x `mod` 2,BBS modulus ((x*x) `mod` modulus))
instance PRNG BBSRand where
    nextB = nextBBSBit


main = do
    print "HER"
