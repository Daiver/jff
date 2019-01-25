#include <stdio.h>

int main()
{
    float f = 1234567.1234567;
    printf("%.3f\n", f);
    f = 1234.1234567;
    printf("%.3f\n", f);
    f = 1234.1;
    printf("%.3f\n", f);
    f = 1234.123456;
    printf("%.*f\n", 4, f);
    return 0;
}
