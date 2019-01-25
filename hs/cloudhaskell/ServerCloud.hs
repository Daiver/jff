module Main where
import qualified Control.Distributed.Backend.P2P  as P2P
import           Control.Distributed.Process      as DP
import           Control.Distributed.Process.Node as DPN

import System.Environment (getArgs)

import Control.Monad
import Control.Monad.Trans (liftIO)
import Control.Concurrent (threadDelay)

main :: IO ()
main = do
  args <- getArgs

  case args of
    host:port:seeds -> P2P.bootstrap host port (map P2P.makeNodeId seeds) initRemoteTable mainProcess
    _ -> putStrLn "Usage: jollycloud addr port [<seed>..]"

mainProcess :: Process ()
mainProcess = do
    spawnLocal logger

    forever $ do
        cmd <- liftIO getLine
        case words cmd of
            ["all"] -> listPeers
            ["in", r] -> listRoom r
            ["join", r] -> joinRoom r
            ["part", r] -> partRoom r
            "tell":r:msg -> tellRoom r (unwords msg)
            _ -> liftIO . putStrLn $ "all | in <r> | join <r> | part <r> | tell <r> <msg>"

logger :: Process ()
logger = do
    unregister "logger"
    getSelfPid >>= register "logger"
    forever $ do
        (time, pid, msg) <- expect :: Process (String, ProcessId, String)
        liftIO $ putStrLn $ time ++ " " ++ show pid ++ " " ++ msg
        return ()

listPeers = P2P.getPeers >>= (liftIO . print)

listRoom r = P2P.getCapable r >>= (liftIO . print)

joinRoom r = do
    pid <- whereis r
    case pid of
        Nothing -> spawnLocal (roomService r) >>= register r
        Just _ -> return ()

partRoom r = do
    pid <- whereis r
    case pid of
        Nothing -> return ()
        Just p -> send p (Nothing :: Maybe String)

tellRoom r msg = P2P.nsendCapable r (Just msg)

roomService :: String -> Process ()
roomService s = do
    msg <- expect :: Process (Maybe String)
    case msg of
        Nothing -> do
            liftIO . putStrLn $ "Leaving: " ++ s
            unregister s
        Just m -> do
            liftIO . putStrLn $ "<" ++ s ++ "> " ++ m
            roomService s
