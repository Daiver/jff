import Control.Monad.Writer
import Control.Monad.Reader

test :: Int -> Writer [String] Int
test x = tell ["Added " ++ show x] >> return x

countDown :: Int -> Writer [String] ()
countDown 0 = tell ["END!"]
countDown x = tell ["is " ++ show x] >> countDown (x - 1) 

test2 :: Reader String String
test2 = do
    name <- ask
    return ("Hello " ++ name)

test3 :: Reader String String
test3 = do
    name <- ask
    return ("By " ++ name)

hello :: Reader String String
hello = do
    name <- ask
    return ("hello, " ++ name ++ "!")
 
bye :: Reader String String
bye = do
    name <- ask
    return ("bye, " ++ name ++ "!")
 
convo :: Reader String String
convo = do
    c1 <- hello
    c2 <- bye
    return $ c1 ++ " " ++ c2

test4 :: Reader Int (Writer [String] Int)
test4 = do
    x <- ask
    return (test =<< test x)


main = do
    print "Start"
    print $ runWriter $ test 10
    print $ runWriter $ test 12 >>= test
    print $ runWriter $ countDown 10
    print $ runReader convo $ "10"
    print $ runWriter . runReader test4 $ 10
