def f1():
    print x

def f2():
    for x in xrange(3):
        f1()

if __name__ == '__main__':
    f2()
    #for x in xrange(3):
    #    f1()
