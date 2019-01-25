{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}

import Network.Transport.TCP (createTransport, defaultTCPParameters)
import Control.Distributed.Process
import Control.Distributed.Process.Node
import Control.Concurrent (threadDelay)
import qualified Data.Map.Strict as M
import Control.Monad
import Data.Maybe
import GHC.Generics (Generic)
import Data.Typeable
import Data.Binary
import Text.Printf

data Request
    = Set ProcessId String Int | Get ProcessId String
        deriving (Show, Eq, Typeable, Generic)

instance Binary Request

data Response
    = Ok | Value (Maybe Int)
        deriving (Show, Eq, Typeable, Generic)

instance Binary Response

serverProc :: M.Map String Int -> Process ()
serverProc m = do
    req <- expect :: Process Request
    case req of
        Set p k v -> do
            send p Ok
            serverProc $ M.insert k v m
        Get p k -> do
            let v = M.lookup k m
            send p $ Value v
            serverProc m

clientProc :: ProcessId -> Process ()
clientProc srv = do
    self <- getSelfPid
    send srv $ Get self "counter"
    Value mv <- expect
    let v = fromMaybe 0 mv
    when (v <= 1000) $ do
        say $ printf "counter = %d" v
        send srv $ Set self "counter" (v+1)
        Ok <- expect
        clientProc srv

serverStart :: Process ProcessId
serverStart = do
    say "Starting server"
    spawnLocal $ serverProc M.empty

clientStartMonitor :: ProcessId -> Process ()
clientStartMonitor srv = do
    say "Starting client"
    pid <- spawnLocal $ clientProc srv
    _ <- monitor pid
    return ()

waitClients :: Int -> Process ()
waitClients n =
    when (n > 0) $ do
        ProcessMonitorNotification{} <- expect
        say "Client terminated"
        waitClients (n-1)

main = do
    Right t <- createTransport "127.0.0.1" "4444" defaultTCPParameters
    node <- newLocalNode t initRemoteTable
    runProcess node $ do
        srv <- serverStart
        let n = 3
        replicateM_ n $ clientStartMonitor srv
        waitClients n
        liftIO $ threadDelay 3000000
