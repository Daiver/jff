{-# LANGUAGE ForeignFunctionInterface #-}
module Sheep(
sheep_with, 
sheep_delete, 
sheep_new, 
sheep_make_sound, 
sheep_shear
)

where

import Types
import Control.Monad

import Foreign
import Foreign.C.String
import Foreign.C.Types

sheep_with :: Int -> (Sheep -> IO a) -> IO a
sheep_with p1 f = do
    obj <- sheep_new p1
    res <- f obj
    sheep_delete obj
    return res

foreign import ccall "Sheep.h Animals_Sheep_delete" c_sheep_delete :: Sheep -> IO ()
sheep_delete :: Sheep -> IO ()
sheep_delete p1 =   c_sheep_delete p1

foreign import ccall "Sheep.h Animals_Sheep_new" c_sheep_new :: CInt -> IO Sheep
sheep_new :: Int -> IO Sheep
sheep_new p1 =   c_sheep_new (fromIntegral p1)

foreign import ccall "Sheep.h Animals_Sheep_make_sound" c_sheep_make_sound :: Sheep -> IO ()
sheep_make_sound :: Sheep -> IO ()
sheep_make_sound p1 =   c_sheep_make_sound p1

foreign import ccall "Sheep.h Animals_Sheep_shear" c_sheep_shear :: Sheep -> IO ()
sheep_shear :: Sheep -> IO ()
sheep_shear p1 =   c_sheep_shear p1

