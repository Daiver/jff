--import Text.XML.HXT.Core
import Network.Curl
import Control.Monad 
import Control.Applicative
import Text.XML.Light

is_image_lin Nothing = False
is_image_link (Just s) = (reverse ".jpg") == (take 4 $ reverse s)

main
    = do
        tmp <- parseXMLDoc . snd <$> curlGetString "http://www.goodfon.ru/" []
        print $ filter (\ x -> is_image_link  x)  <$> map (findAttr (QName "src" Nothing Nothing)) <$> findElements (QName "img" Nothing Nothing) <$> tmp
