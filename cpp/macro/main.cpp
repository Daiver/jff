#include <stdio.h>

/*
#define TYPEOF(T) \
    template<> \

    struct<>
*/

template <typename T>
struct JokeS{
    typedef T type;
};

template<typename T>
JokeS<T> getJoke(T x)
{
    return JokeS<T>();
}

#ifdef __GNUG__
#define x 10000
#else
#define x 1
#endif

int main()
{
    //int i = (getJoke(10)::type) 1000.0;
    __typeof__(10) i = 10;
    printf("%d", x);
    return 0;
}
