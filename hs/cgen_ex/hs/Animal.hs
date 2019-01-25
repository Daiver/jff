{-# LANGUAGE ForeignFunctionInterface #-}
module Animal(
animal_delete, 
animal_make_sound, 
animal_get_age, 
animal_increment_age
)

where

import Types
import Control.Monad

import Foreign
import Foreign.C.String
import Foreign.C.Types

foreign import ccall "Animal.h Animals_Animal_delete" c_animal_delete :: Animal -> IO ()
animal_delete :: Animal -> IO ()
animal_delete p1 =   c_animal_delete p1

foreign import ccall "Animal.h Animals_Animal_make_sound" c_animal_make_sound :: Animal -> IO ()
animal_make_sound :: Animal -> IO ()
animal_make_sound p1 =   c_animal_make_sound p1

foreign import ccall "Animal.h Animals_Animal_get_age" c_animal_get_age :: Animal -> IO CInt
animal_get_age :: Animal -> IO Int
animal_get_age p1 =  liftM fromIntegral $  c_animal_get_age p1

foreign import ccall "Animal.h Animals_Animal_increment_age" c_animal_increment_age :: Animal -> IO ()
animal_increment_age :: Animal -> IO ()
animal_increment_age p1 =   c_animal_increment_age p1

