import Text.ParserCombinators.Parsec --hiding(spaces)
import System.Environment
import Control.Monad
import Data.Char
import Data.List

data LispVal = LAtom String | LString String | LInt Int | LV [LispVal] | LVector [LispVal]
    deriving Show

--spaces :: Parser ()
--spaces = skipMany1 space

parseString :: Parser LispVal
parseString = do
    char '"'
    x <- many (noneOf "\"")
    char '"'
    return $ LString x

symbol :: Parser Char
symbol = oneOf "!#$%&|*+-/:<=>?@^_~"

parseNumber :: Parser LispVal
parseNumber = liftM (LInt . read) $ many1 digit

parseVector :: Parser LispVal
parseVector = do
    char '['
    items <- many (spaces >> parseExpr)
    spaces
    char ']'
    spaces
    return $ LVector items

parseLAtom :: Parser LispVal
parseLAtom = spaces >> do
    first <- letter <|> symbol
    rest  <- many (letter <|> digit <|> symbol)
    return . LAtom $ first:rest

parseExpr :: Parser LispVal
parseExpr = parseLAtom
        <|> lispVal
        <|> parseString
        <|> parseNumber
        <|> parseVector

lispVal = do
    char '('
    els <- many (spaces >> parseExpr)
    spaces
    char ')'
    return $ LV els

main = do
    print "HI"
    text <- readFile "to_l_parse"
    case (parse ( parseExpr) "some text" text) of
        Left err -> print "err" >> print err
        Right x -> print x


