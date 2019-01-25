#include <stdio.h>

float cosinus(float x)
{
    float xSquared = x*x;
    //1 -  x^2/2! + x^4/4! - x^6/6!
    const int maxIter = 100;
    float elem = 1;
    float sum = 0;
    for(int i = 1; i < maxIter; ++i){
        sum += elem;
        elem *= -1;
        elem *= xSquared/((2*i - 1)*(2*i));
    }
    return sum;
}

int main()
{
    printf("%f\n", cosinus(3.14/2*3));
    return 0;
}
