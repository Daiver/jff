--import Text.XML.HXT.Core
import Network.Curl
import Control.Monad 
import Control.Applicative
import Text.XML.Light
import Text.Format

main = do
    let url = format "http://google.com/search?hl=en&as_q={0}&num={1}&as_qdr={2}" ["haskell", show 10, ""]
    print url
    tmp <- parseXMLDoc . snd <$> curlGetString url []
    print tmp
    --let res = filter (\ x -> is_image_link  x)  <$> map (findAttr (QName "src" Nothing Nothing)) <$> findElements (QName "img" Nothing Nothing) <$> tmp
    --print res
