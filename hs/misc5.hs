import Control.Monad.Cont
import Control.Monad (when)


inc_Cont :: Int -> Cont r Int
inc_Cont x = return (x + 1)

square_Cont :: Int -> Cont r Int
square_Cont x = return (x*x)

f_C :: Int -> Cont r Int
f_C x = (\x -> return (x)) =<< square_Cont =<< inc_Cont =<< inc_Cont x

f_C2 :: Int -> Cont r Int
f_C2 x = do
    a <- inc_Cont x
    b <- square_Cont a
    return b

hehe :: Int -> Cont r String
hehe n = callCC $ \exit -> do
    let fac = product [1..n]
    when (n > 7) $ exit "OVER 9000"
    return $ show fac

tmp :: (a -> b) -> a -> b
tmp f x = f x

main = do
    runCont (f_C 10) (putStrLn . show)
    print $ runCont (f_C2 10) (+1)
    runCont (hehe 8) putStrLn
    print $ (tmp map (+1)) [1,2,3]
