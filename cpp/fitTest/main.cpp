#include <iostream>

#include "fit/infix.h"
#include "fit/compose.h"
#include "fit/lambda.h"
#include "fit/placeholders.h"

using namespace fit;

struct increment
{
    template<class T>
    T operator()(T x) const
    {
        return x + 1;
    }
};

struct decrement
{
    template<class T>
    T operator()(T x) const
    {
        return x - 1;
    }
};


int main()
{
    int r = compose(increment(), decrement(), increment())(3);
    auto cmps = infix([](auto x, auto y){return compose(x, y);});
    //auto cmps = infix(compose);
    int k = (increment() <cmps> increment())(5);
    int k2 = (([](auto x) {return x + 10;}) <cmps> increment() <cmps> (_ + 50))(5);
    std::cout << r << " " << k << " " << k2;
    return 0;
}
