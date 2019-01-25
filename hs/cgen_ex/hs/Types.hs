module Types
where

import Foreign
import Foreign.C.String
import Foreign.C.Types

type CBool = CChar -- correct?

newtype Animal = Animal (Ptr Animal) -- nullary data type
newtype Dog = Dog (Ptr Dog) -- nullary data type
newtype Sheep = Sheep (Ptr Sheep) -- nullary data type
newtype Void = Void (Ptr Void) -- nullary data type

class CAnimal a where
  toAnimal :: a -> Animal

instance CAnimal Dog where
  toAnimal (Dog p) = Animal (castPtr p)

instance CAnimal Sheep where
  toAnimal (Sheep p) = Animal (castPtr p)

