module Data.SimpleJSON where

import Data.List (intercalate)
import Data.Char (isDigit)

data JValue = JString String
 | JNumber Double
 | JBool Bool
 | JNull
 | JObject [(String, JValue)]
 | JArray [JValue]
 deriving (Eq, Ord, Show)

bracketsTest str 
    | (h, l) == ('"', '"') = JString suffix
    | isNum s = JNumber (read s)
    | otherwise = JNull
    where
        dropWS = dropWhile (==' ')
        isNum = all isDigit
        s = reverse . dropWS . reverse . dropWS $ str
        h = head s
        l = last s
        suffix = tail . init $ s

instance Read JValue where
    readsPrec _ s = [(bracketsTest s, s)]

getString (JString s) = Just s
getString _ = Nothing

getInt (JNumber n) = Just (truncate n)
getInt _ = Nothing

getDouble (JNumber n) = Just n
getDouble _ = Nothing

getBool (JBool b) = Just b
getBool _ = Nothing

getObject (JObject o) = Just o
getObject _ = Nothing

getArray (JArray a) = Just a
getArray _ = Nothing

isNull v = v == JNull

renderJValue :: JValue -> String

renderJValue (JString s) = show s

renderJValue (JNumber n) = show n

renderJValue (JBool True) = "true"

renderJValue (JBool False) = "false"

renderJValue JNull = "null"

renderJValue (JObject o) = "{" ++ pairs o ++ "}"
 where pairs [] = ""
       pairs ps = intercalate ", " (map renderPair ps)
       renderPair (k,v) = show k ++ ": " ++ renderJValue v

renderJValue (JArray a) = "[" ++ values a ++ "]"
 where values [] = ""
       values vs = intercalate ", " (map renderJValue vs)
