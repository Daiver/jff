
data Some = S {x::Bool, y::Bool, z::Bool}

funcs = [x, y, z]

main = do
    let s = S True False True
    print $ map ($s) funcs
