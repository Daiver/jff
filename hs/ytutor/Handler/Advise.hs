module Handler.Advise where

import Import

getAdviseR :: Handler Html
getAdviseR = do
    let theAdvise = "Do not worry" :: String
    defaultLayout $(widgetFile "advise")

postAdviseR :: Handler Html
postAdviseR = error "Not yet implemented: postAdviseR"
