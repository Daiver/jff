#include <stdio.h>

template<long long X>
inline long long fib()
{
    return fib<X - 1>() + fib<X - 2>();
}

template<>
inline long long fib<1>()
{
    return 1;
}

template<>
inline long long fib<2>()
{
    return 1;
}


int main()
{
    printf("%lld\n", fib<401>());
    return 0;
}
