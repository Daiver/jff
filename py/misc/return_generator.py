
def f1():
    for x in xrange(5):
        print 'generated', x
        yield x

def f2():
    yield f1()

for x in f2():
    print 'used', x
