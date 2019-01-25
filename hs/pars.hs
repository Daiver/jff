import Text.ParserCombinators.Parsec
import System.Environment
import Data.Char
import Data.List

type Entry = (String, String)
type Section = (String, [Entry])
type IniData = [Section]

inidata = spaces >> many section >>= return

section = do
    char '['
    name <- ident
    char ']'
    stringSpaces
    char '\n'
    spaces 
    el <- many entry
    return (name, el)

entry = do
    k <- ident
    char '='
    stringSpaces
    v <- value
    spaces
    return (k, v)

ident = many1 (letter <|> digit <|> oneOf " _.,:)_{}-#@&*|") >>= return . trim

value = many (noneOf "\n") >>= return . trim

stringSpaces = many (char ' ' <|> char '\t')

trim = f . f
    where f = reverse . dropWhile isSpace

split delim = foldr f [[]]
    where 
        f x rest@(r:rs)
            | x == delim = [delim] : rest
            | otherwise = (x:r) : rs

removeComment = foldr (++) [] . filter comment . split '\n'
    where   
        comment [] = False
        comment (x:_) = (x /= ';') -- && (x /= '\n')

findValue ini s p = do
    el <- find (\x -> fst x == s) ini
    v  <- find (\x -> fst x == p) (snd el)
    return $ snd $ v

main = do
    args <- getArgs
    prog <- getProgName
    if(length args) /= 3
        then print "NOPE"
        else do
            file <- readFile $ head args
            [s,p] <- return $ tail args
            lns <- return (removeComment file)
            case (parse inidata "some text" lns) of
                Left err -> print "err" >> print err
                Right x -> case (findValue x s p) of
                    Just x -> print x
                    Nothing -> print "No param"




