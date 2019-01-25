#include <stdio.h>

template<int X>
int fact()
{
    return fact<X - 1>() * X;
}

template<>
int fact<1>()
{
    return 1;
}

int main()
{
    printf("%d\n", fact<100>());
    return 0;
}
