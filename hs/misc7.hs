
{-(+) :: (Num a, Num b) => (a, b) -> (a, b) -> (a, b)-}
{-(+) (x1, y1) (x2, y2) = (x1 Prelude.+ x2, y1 Prelude.+ y2)-}

type Point a = (a, a)

{-instance Num (a, a) where-}
    {-(+) (x1, y1) (x2, y2) = (x1 Prelude.+ x2, y1 Prelude.+ y2)-}

main = do
    print "Hi"
    {-print $ (1, 2) Main.+ (3, 4)-}
