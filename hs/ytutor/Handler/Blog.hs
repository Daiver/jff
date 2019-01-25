module Handler.Blog where

import Data.Text as T
import Import
import qualified Data.List as L
import Yesod.Form.Nic (YesodNic, nicHtmlField)

instance YesodNic App


entryForm :: Form Article
entryForm = renderDivs $ Article
    <$> areq   textField "Title" Nothing
    <*> areq   nicHtmlField "Content" Nothing

data ArticleData = ArticleData {
        articleDataTitle   :: Text,
        articleDataContent :: Html,
        articleDataTagList :: Text
    } 

articleForm :: Form ArticleData
articleForm = renderDivs $ ArticleData
    <$> areq   textField "Title" Nothing
    <*> areq   nicHtmlField "Content" Nothing
    <*> areq   textField "TagList" Nothing

setTags aId tags = runDB $ do
    existedTags <- selectList [TagName <-. tags] []
    _ <- insert $ TagArticle (TagId . L.head $ existedTags) aId
    --_ <- mapM (insert . flip TagArticle articleId . tagName) existedTags
    --let words = L.map TagName existedTags
    --let (exWords, newWords) = L.partition (\x -> )
    return $ existedTags

-- The view showing the list of articles
getBlogR :: Handler Html
getBlogR = do
    -- Get the list of articles inside the database.
    articles <- runDB $ selectList [] [Desc ArticleTitle]
    -- We'll need the two "objects": articleWidget and enctype
    -- to construct the form (see templates/articles.hamlet).
    (articleWidget, enctype) <- generateFormPost entryForm
    (articleDataWidget, enctype2) <- generateFormPost articleForm
    defaultLayout $ do
        $(widgetFile "articles")

postBlogR :: Handler Html
postBlogR = do
    --((res,articleWidget),enctype) <- runFormPost entryForm
    ((res,articleDataWidget),enctype) <- runFormPost articleForm

    case res of
         FormSuccess dt -> do 
            articleId <- runDB $ insert $ Article (articleDataTitle dt) (articleDataContent dt)
            tagArticleId <- setTags articleId (T.words . articleDataTagList $ dt)
            -- $(logInfo) . T.pack . mconcat . L.map TagName tagArticleId
            -- $(logInfo) $ mconcat [">>>", T.pack $ show $ L.length tagArticleId]
            setMessage $ toHtml $ (articleDataTitle dt) <> " created"
            redirect $ ArticleR articleId
            
         _ -> defaultLayout $ do
                setTitle "Please correct your entry form"
                $(widgetFile "articleAddError")


    {-case res of
         FormSuccess article -> do
            articleId <- runDB $ insert article
            setMessage $ toHtml $ (articleTitle article) <> " created"
            redirect $ ArticleR articleId
         _ -> defaultLayout $ do
                setTitle "Please correct your entry form"
                $(widgetFile "articleAddError")
    -}

--getBlogR :: Handler Html
--getBlogR = error "Not yet implemented: getBlogR"

--postBlogR :: Handler Html
--postBlogR = error "Not yet implemented: postBlogR"
