module Handler.Article where

import Import

getArticleR :: ArticleId -> Handler Html
getArticleR articleId = do
    article <- runDB $ get404 articleId
    defaultLayout $ do
        setTitle $ toHtml $ articleTitle article
        $(widgetFile "article")

--getArticleR :: ArticleId -> Handler Html
--getArticleR = error "Not yet implemented: getArticleR"
