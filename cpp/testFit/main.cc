
#include <stdio.h>
#include <iostream>

#include "../Fit/fit/lambda.h"
#include "../Fit/fit/infix.h"

int main()
{
    auto plus = fit::infix([](int x, int y)
    {
        return x + y;
    });
    std::cout << (1 <plus> 10);
    return 0;
}
