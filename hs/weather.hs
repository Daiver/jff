import Network.Curl
import Control.Monad 
import Control.Applicative
import Text.XML.Light
import Data.List.Split(splitOn)

simplify = foldl (++) []

city = "Voronez,ru"
url = "http://api.openweathermap.org/data/2.5/weather?q=" ++ city

temp_template = "\"temp\""
desc_template = "\"description\""

findData template s = foldl (++) "" proclst
    where 
        lst = filter (\x -> (take (length template) x) == template) . simplify . map (splitOn "{")  . splitOn "," $ s
        proclst = map (tail . snd . break (==':')) lst

processTemp :: [Char] -> Float
processTemp t = (read t) - 273

main = do
        putStr . (\x -> "T:" ++ city ++ "-" ++ (show x)) . processTemp . findData temp_template . snd =<< curlGetString url []


