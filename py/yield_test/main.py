
def func(n):
    for i in xrange(n):
        print 'current on', i
        yield (i, i**2)

gen = func(10)
for x, y in gen:
    print x, '/', y
