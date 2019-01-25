import qualified Data.Map as M
import Data.Map 
import Data.Maybe
import Control.Monad

data DValue = DString String | DBool Bool | DDict (M.Map String DValue)
    deriving Show

asString :: DValue -> Maybe String
asString (DString s) = Just s
asString _           = Nothing

asDict :: DValue -> Maybe (M.Map String DValue)
asDict (DDict d) = Just d
asDict _         = Nothing


main = do
    let example = (M.fromList [
                            ("picture",
                                DDict ( M.fromList [
                                    ("data", DDict (M.fromList [
                                            ("url", DString "http://Nothing")
                                        ])
                                    )
                                ])
                            )
                        ])

    let x = M.lookup "picture" example >>= asDict >>= M.lookup "data" >>= asDict >>= M.lookup "url" >>= asString

    let y = M.lookup "picture" example >>= asDict >>= M.lookup "data" >>= asDict >>= M.lookup "url2" >>= asString
    print x
    print y
