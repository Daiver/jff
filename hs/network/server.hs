import Network.Socket hiding (send, recv)
import Control.Concurrent (forkIO)
import qualified Data.ByteString.Char8 as B8
import Network.Socket.ByteString (send, recv)

import System.Environment (getArgs)

--server :: PortNumber -> IO ()
server port = withSocketsDo $ do
                sock <- socket AF_INET Stream defaultProtocol
                bindSocket sock (SockAddrInet port 0)
-- Слушаем сокет. 
-- Максимальное кол-во клиентов для подключения - 5.
                listen sock 5
-- Запускаем наш Socket Handler.
                sockHandler sock                 
                sClose sock

sockHandler :: Socket -> IO ()
sockHandler sock = do
-- Принимаем входящее подключение.
  (sockh, _) <- accept sock
-- В отдельном потоке получаем сообщения от клиента.
  forkIO $ putStrLn "Client connected!" >> receiveMessage sockh
  sockHandler sock

receiveMessage :: Socket -> IO ()
receiveMessage sockh = do
  msg <- recv sockh 10 -- Получаем только 10 байт.
  B8.putStrLn msg -- Выводим их.
-- Если сообщение было пусто или оно равно "q" (quit)
  if msg == B8.pack "q" || B8.null msg
-- Закрываем соединение с клиентом.
  then sClose sockh >> putStrLn "Client disconnected"
  else send sockh (B8.reverse msg) >> receiveMessage sockh -- Или получаем следующее сообщение.

main = do
    server 3000
