import Text.ParserCombinators.Parsec --hiding(spaces)
import System.Environment
import Control.Monad
import Data.Char
import Data.List

main = do
    runTests
    --let text = "\nf = (a b) -> a + b;\nmain = () -> print (f 10 3)"
    --print text
    text <- readFile "ptest.l"
    prep <- readFile "prep.js"
    --case (parse (spaces >> parseExpr) "some text" text) of
    case (parse (spaces >> parseManyExpr) "some text" text) of
        Left err -> print "err" >> print err
        Right x -> mapM print x >> (writeFile "tmp.js" $ write2JS prep x)
        --Right x -> print x >> (print $ write2JS prep x)
        --Right x -> print x >> (print $ executeAst x)


data FuncExpr = FAtom String | FInt Int | FBinOp FuncExpr FuncExpr FuncExpr
                | FApply FuncExpr [FuncExpr] | FDef [FuncExpr] FuncExpr
     deriving (Eq, Show)

data GlobalAssign = GlobalAssign FuncExpr FuncExpr deriving Show

--write2JS :: String -> [FuncExpr] -> String
write2JS :: String -> [GlobalAssign] -> String
write2JS prep exprs = 
    prep ++ "\n" ++ (concat . map procOne $ exprs)  ++ "\nmain()\n"
    where 
        procOne (GlobalAssign (FAtom name) body) = 
            "var " ++ getOp name ++ " = " ++ trans2JS body ++ " \n"

getOp :: String -> String
getOp = concat . map get 
    where
        get '+' = "op_Plus" 
        get '-' = "op_Minus" 
        get '*' = "op_Mul" 
        get '/' = "op_Div" 
        get '.' = "op_Dot" 
        get c = [c]

trans2JS :: FuncExpr -> String

trans2JS (FAtom atm) = getOp atm

trans2JS (FInt i) = show i
trans2JS (FBinOp f x y) = trans2JS f ++ "(" ++ trans2JS x ++ ", " ++ trans2JS y ++ ")"
trans2JS (FApply f args) = trans2JS f ++ "(" ++ 
    (concat . map (\x -> trans2JS x ++ ",") $ init args) ++ (trans2JS . last $ args) ++ ")"

trans2JS (FDef args body) = hd ++ "{ return " ++ trans2JS body ++ ";});\n"
    where 
        --hd = "function (" ++ (joins ","  ["1", "2"]) ++ ")"
        hd = "(function (" ++ (joins "," . map showAtom $ args) ++ ")"
        showAtom (FAtom a) = a

symbol :: Parser Char
symbol = oneOf symbols_list

symbols_list ="!#$%&|*+-/:<=>?@^_~."

parseNumber :: Parser FuncExpr
parseNumber = liftM (FInt . read) $ many1 digit

parseIdent = do
    first <- letter 
    rest  <- many (letter <|> digit )
    return $ first : rest

parseOpIdent = do
    char '('
    ident <- many1 symbol
    char ')'
    return ident

parseAtom :: Parser FuncExpr
parseAtom = do
    --first <- letter 
    --rest  <- many (letter <|> digit )
    ident <- try parseIdent <|> parseOpIdent
    return . FAtom $ ident --first : rest
 
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
    op    <- (try $ many1 symbol ) <|> (parseSpec)
    spaces 
    rest <- parseExpr
    spaces
    return $ FBinOp (FAtom op) first rest
    where 
        parseSpec = do
            char '`'
            ident <- parseIdent
            char '`'
            return ident

parseDef :: Parser FuncExpr
parseDef = do
    --char '\\'
    char '('
    args <- many (spaces >> parseAtom) -- `sepBy` (oneOf ",")
    char ')' 
    spaces
    char '-' >> char '>' >> spaces
    body <- parseExpr
    --body <- parseSimple
    return $ FDef args body
    


parseSimple = try parseApply <|> try parseAtom <|> try parseBrackets <|> try parseNumber 
parseExpr = try parseDef <|> try parseBinOp <|> parseSimple

parseAssign :: Parser GlobalAssign
parseAssign = do
    name <- parseAtom
    spaces
    char '='
    spaces
    ex <- parseExpr
    spaces 
    return $ GlobalAssign name ex

--parseManyExpr :: Parser [FuncExpr]
parseManyExpr :: Parser [GlobalAssign]
parseManyExpr =  parseAssign `sepBy` (many1 $ oneOf "\n;")
--parseManyExpr =  parseExpr `sepBy` (char '\n')
    --char ';'
    --expr2 <- parseExpr --`sepBy` (char '\n')
    --return [expr1 , expr2]

executeAst :: FuncExpr -> Float

executeAst (FInt i) = fromIntegral i

executeAst (FBinOp (FAtom c) x y) = op (executeAst x) (executeAst y)
    where 
        op = case c of
            "+" -> (+)
            "-" -> (-)
            "*" -> (*)
            "/" -> (/)


joins :: String -> [String] -> String
joins _ [] = ""
joins sep (x:xs) = x ++ (concat . map (\y -> sep ++ y) $ xs)

runTest text expr = 
    case (parse (spaces >> parseExpr) "some text" text) of
        Left err -> print "err" >> print err 
        Right x -> do
            if (x /= expr) then print "Err" >> print x
            else return ()
        --Right x -> print x >> (print $ write2JS prep x)
        --Right x -> print x >> (print $ executeAst x)

runTests = do
    --let text = "\\a b -> (a + b)"
    let text = "(a b) -> (a + b)"
    runTest text (FDef [FAtom "a",FAtom "b"] (FBinOp (FAtom "+") (FAtom "a") (FAtom "b")))
    let text = "(a b) -> a + b"
    runTest text (FDef [FAtom "a",FAtom "b"] (FBinOp (FAtom "+") (FAtom "a") (FAtom "b")))
    let text = "(a b) -> a b"
    runTest text (FDef [FAtom "a",FAtom "b"] (FApply (FAtom "a") [(FAtom "b")]))
    let text = "1 + 1"
    runTest text (FBinOp (FAtom "+") (FInt 1) (FInt 1))
    let text = "1 + 1 + 1"
    runTest text (FBinOp 
            (FAtom "+") (FInt 1) (FBinOp (FAtom "+") (FInt 1) (FInt 1)))
    let text = "1 + (1 - 1)"
    runTest text (
            FBinOp (FAtom "+") (FInt 1) (FBinOp (FAtom "-") (FInt 1) (FInt 1)))
    let text = "f 10"
    runTest text (FApply (FAtom "f") [FInt 10])
    let text = "10 `ff` 100"
    runTest text (FBinOp (FAtom "ff") (FInt 10) (FInt 100))
    let text = "10 `ff` (f 10)"
    runTest text (FBinOp (FAtom "ff") (FInt 10) (FApply (FAtom "f") [FInt 10]))
    let text = "(*) 10 (f 10)"
    runTest text (FApply (FAtom "*") [(FInt 10), (FApply (FAtom "f") [FInt 10])])
    let text = "(*) (f 10) (f 10)"
    runTest text (FApply (FAtom "*") [(FApply (FAtom "f") [(FInt 10)]), (FApply (FAtom "f") [FInt 10])])


