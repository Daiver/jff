{-# LANGUAGE ForeignFunctionInterface #-}
module Dog(
dog_with, 
dog_delete, 
dog_new, 
dog_make_sound
)

where

import Types
import Control.Monad

import Foreign
import Foreign.C.String
import Foreign.C.Types

dog_with :: (Dog -> IO a) -> IO a
dog_with  f = do
    obj <- dog_new 
    res <- f obj
    dog_delete obj
    return res

foreign import ccall "Dog.h Animals_Dog_delete" c_dog_delete :: Dog -> IO ()
dog_delete :: Dog -> IO ()
dog_delete p1 =   c_dog_delete p1

foreign import ccall "Dog.h Animals_Dog_new" c_dog_new :: IO Dog
dog_new :: IO Dog
dog_new  =   c_dog_new 

foreign import ccall "Dog.h Animals_Dog_make_sound" c_dog_make_sound :: Dog -> IO ()
dog_make_sound :: Dog -> IO ()
dog_make_sound p1 =   c_dog_make_sound p1

