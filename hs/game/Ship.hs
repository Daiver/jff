module Ship where

import GameTypes
import GHC.Word 

data Hull = Hull {
    size  :: SpaceSize,
    value :: Word
}

data Ship = Ship {
    so   :: SpaceObject,
    hull :: Hull
}
