import Text.ParserCombinators.Parsec --hiding(spaces)
import System.Environment
import Control.Monad
import Data.Char
import Data.List

main = do
    let text = "1 + 1"
    let text = "1 + 1 + 1"
    let text = "1 + (1 - 1)"
    let text = "f 1 1 + a + f (x c d) (f 1 2 3)"
    let text = "10 + 1 * 2"
    print text
    case (parse (spaces >> parseExpr) "some text" text) of
        Left err -> print "err" >> print err
        Right x -> print x >> (print $ executeAst x)


data FuncExpr = FAtom String | FInt Int | FBinOp FuncExpr FuncExpr FuncExpr
                | FApply FuncExpr [FuncExpr]
     deriving Show

symbol :: Parser Char
symbol = oneOf "!#$%&|*+-/:<=>?@^_~"

parseNumber :: Parser FuncExpr
parseNumber = liftM (FInt . read) $ many1 digit

parseAtom :: Parser FuncExpr
parseAtom = do
    first <- letter 
    rest  <- many (letter <|> digit )
    return . FAtom $ first : rest
 
parseApply :: Parser FuncExpr
parseApply = do
    first <- parseAtom
    char ' '
    spaces
    rest <- many1 (spaces >> parseSimple)
    return $ FApply first rest

parseBrackets :: Parser FuncExpr
parseBrackets = do
    char '('
    exp <- parseExpr
    char ')'
    return exp

parseBinOp :: Parser FuncExpr
parseBinOp = do
    first <- parseSimple
    spaces
    op    <- many1 symbol
    spaces 
    rest <- parseExpr
    spaces
    return $ FBinOp (FAtom op) first rest


parseSimple = try parseApply <|> try parseAtom <|> try parseBrackets <|> try parseNumber 
parseExpr = try parseBinOp <|> parseSimple

executeAst :: FuncExpr -> Float

executeAst (FInt i) = fromIntegral i

executeAst (FBinOp (FAtom c) x y) = op (executeAst x) (executeAst y)
    where 
        op = case c of
            "+" -> (+)
            "-" -> (-)
            "*" -> (*)
            "/" -> (/)
