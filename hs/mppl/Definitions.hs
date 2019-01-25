module Definitions where

data Function a = Function String ([a] -> a)
    | BinaryOperator String (a -> a -> a) 
    | UnaryOperator String (a -> a) 

instance Eq (Function a) where
    (Function s1 _) == (Function s2 _) = s1 == s2
    (BinaryOperator s1 _) == (BinaryOperator s2 _) = s1 == s2
    (UnaryOperator s1 _) == (UnaryOperator s2 _) = s1 == s2

instance Show (Function a) where    
    show (Function s _) = "Function " ++ s
    show (BinaryOperator s _) = "BinaryOperator " ++ s
    show (UnaryOperator s _) = "UnaryOperator " ++ s

data Operation a = 
      Variable String
    | Constant a
    | Operation (Function a) [Operation a]
    deriving Eq

instance (Show a) => Show (Operation a) where    
    show (Operation (Function s _) []) = concat [s, "()"]
    show (Operation (UnaryOperator s _) []) = concat [s, "()"]
    show (Operation (Function s _) l) = concat [s, "(", 
            concat . init . init . map (\x -> show x ++ ", ") $ l, ")"]
    show (Operation (BinaryOperator s _) (a:b:_)) = concat [
            "(", show a, ")", s, "(", show b, ")"]
    show (Operation (UnaryOperator s _) (a:_)) = concat [s, "(", show a, ")"]
    show (Constant x) = show x
    show (Variable x) = x

getFunctionName :: Function a -> String
getFunctionName (Function s _)       = s
getFunctionName (BinaryOperator s _) = s
getFunctionName (UnaryOperator s _)  = s
