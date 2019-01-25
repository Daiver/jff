{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE EmptyDataDecls       #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE QuasiQuotes          #-}
{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

import qualified Web.Scotty as S
import Data.Monoid (mconcat)
import System.Random (newStdGen, randomRs)
import qualified Data.Map as M
import Text.Blaze.Html5.Attributes
import Text.Blaze.Html.Renderer.Text
import Data.Text.Lazy(pack)
import Database.Persist.Sqlite (SqlPersist, withSqliteConn, runSqlConn, runMigration, runSqlite)
import Control.Monad.Trans.Resource (runResourceT, ResourceT)
import Control.Monad.IO.Class (liftIO)
--import Database.Persist.GenericSql
import Database.Persist
import Database.Persist.TH
import Database.Persist.Sql

share [mkPersist sqlSettings, mkMigrate "migrateAll"] [persistLowerCase|
ShortenURL
    url String
    deriving Show
|]

extractKey :: KeyBackend backend entity -> String
extractKey = extractKey' . unKey
    where extractKey' (PersistInt64 k) = show k
          extractKey' _ = ""

runDb = runSqlite "dev.sqlite3"

keyFromParam = Key . PersistInt64 . fromIntegral . read 

replaceParamInTemplate params template = undefined
    where 
        

main = do
    runDb $ runMigration migrateAll
    S.scotty 3000 $ do
        S.get "/static/:fname" $ S.param "fname" >>= (\f -> S.file ("static/" ++ f))
        S.get "/tmp/" $ do
            S.file "tmp.html"
        S.post "/" $ do
            tmp <- S.param "b"
            S.html $ mconcat ["<h1>", tmp ,"</h1>"]

        S.post "/shorten/" $ do
            url <- S.param "url" 
            suId <- liftIO $ runDb $ insert $ ShortenURL url
            let value = pack . extractKey $ suId
            S.html $ mconcat ["<h3 align=center> link <a style='color:blue' href='/shorten/", value,"/' >http://162.243.246.162/shorten/", value, "/ </a></h3>"]
        S.get "/shorten/" $ do
            S.file "shorten.html"
            --S.html $ "<form method=POST action='/shorten' ><input type=edit value=http:// name=url></input><input type=submit></input>"
        S.get "/shorten/:suId" $ do
            id <- S.param "suId"
            li <- liftIO $ runDb $ selectList [ShortenURLId ==. keyFromParam id] [LimitTo 1]
            let value = entityVal . head $ li
            S.redirect $ pack $ shortenURLUrl value
            --liftIO $ print $ head (shortenURLUrl value)
            --liftIO $ print $ shortenURLUrl value
