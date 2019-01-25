def compose(f, g):
    return lambda x:f(g(x))

def foldl(f, init, list):
    for x in list:
        init = f(init, x)
    return init

def pipline(list):
    return foldl(compose, lambda x: x, reversed(list))

def apply(f, x):
    return lambda y: f(x, y)

print pipline([
        apply(map, lambda x: x + 1),
        apply(filter, lambda x: x % 2 == 0),
        apply(map, lambda x: x ** 2),
        sum
    ])([1,2,3,4,5,6,7,8])

