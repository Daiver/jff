
class MyClass1 a where
    pri :: a -> String

class MyClass2 a where
    pri :: a -> String

data DT = DT
instance MyClass1 DT where
    pri DT = "Hi"

main = do
    print ""
