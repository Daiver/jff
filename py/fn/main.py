from funcky import _1, _2, _3, _4, _5

if __name__ == '__main__':
    print _1(1)
    print _2(54, 9)
    print (_1 + _2)(1, 2)
    print (_1 - _2 + _1)(6, 2)
    print (_1 + _2 + _1)(6)(5)
    print (_1 + _2 + _4 + _3 + _1)('!!!')("Hello", "World", " ")
    print (_1 < _2)(1, 2)
    print filter(0 == _1 % 2 , xrange(1, 6))
    # print (_1 * _2 + float(_3))(2, 3, 4)
