
def foo(i):
    for i in xrange(i):
        yield i

for x in foo(12):
    print x
