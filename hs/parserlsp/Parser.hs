module Parser where

import Control.Monad
import Control.Monad.Error
import Data.Maybe
import Data.Char

data Parser t a = Parser ([t] -> Maybe (a, [t]))

instance Monad (Parser t) where
    return v = Parser $ \inp -> Just (v, inp)
    Parser p >>= f = Parser $ \inp -> case p inp of
                        Just (v, out) -> papply (f v) out
                        Nothing -> Nothing

instance MonadPlus (Parser t) where
    mzero = Parser $ \_ -> Nothing
    mplus (Parser p) (Parser q) = Parser $ \inp -> case p inp of
                                            Nothing -> q inp
                                            Just(v, out) -> Just(v, out)

papply (Parser p) inp = p inp

(<|>) :: (MonadPlus m) => m a -> m a -> m a
(<|>) = mplus

item :: Parser t t
item = Parser $ \(x:xs) -> Just(x, xs)

sat :: (t -> Bool) -> Parser t t
sat p = do
    x <- item
    if p x then
        return x
    else
        mzero

char :: Char -> Parser Char Char
char x = sat (==x)

digit :: Parser Char Char
digit = sat isDigit

many, many1 :: Parser t a -> Parser t [a]
many p  = many1 p <|> mzero
many1 p = do
    v <- p
    vs <- many p
    return (v:vs)
