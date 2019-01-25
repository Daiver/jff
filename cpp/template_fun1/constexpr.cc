#include <stdio.h>

constexpr unsigned int factorial(unsigned int i)
{
  return i > 0 ? i*factorial(i-1) : 1;
}

int main()
{
    int x = factorial(500);
    printf("%d\n", x);
    return 0;
}
