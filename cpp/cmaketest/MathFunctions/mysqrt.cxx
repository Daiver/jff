#include "MathFunctions.h"

#include <math.h>
#include <stdio.h>

double mysqrt(double x)
{
    printf("Use mysqrt\n");
    if(x < 0)
        return 0;
    return sqrt(x);
}
