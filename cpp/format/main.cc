#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void){

    float f = 1234.05678900;
    char s[100];
    int decimals;

    decimals = 10;
    sprintf(s,"%.*g", decimals, ((int)(pow(10, decimals)*(fabs(f) - abs((int)f)) +0.5))/pow(10,decimals));
    printf("10 decimals: %d%s\n", (int)f, s+1);

    decimals = 3;
    sprintf(s,"%.*g", decimals, ((int)(pow(10, decimals)*(fabs(f) - abs((int)f)) +0.5))/pow(10,decimals));
    printf(" 3 decimals: %d%s\n", (int)f, s+1);

    f = -f;
    decimals = 10;
    sprintf(s,"%.*g", decimals, ((int)(pow(10, decimals)*(fabs(f) - abs((int)f)) +0.5))/pow(10,decimals));
    printf(" negative 10: %d%s\n", (int)f, s+1);

    decimals = 3;
    sprintf(s,"%.*g", decimals, ((int)(pow(10, decimals)*(fabs(f) - abs((int)f)) +0.5))/pow(10,decimals));
    printf(" negative  3: %d%s\n", (int)f, s+1);

    printf("\n");

    decimals = 2;
    f = 1.012;
    sprintf(s,"%.*g", decimals, ((int)(pow(10, decimals)*(fabs(f) - abs((int)f)) +0.5))/pow(10,decimals));
    printf(" additional : %d%s\n", (int)f, s+1);


    decimals = 6;
    f = 1.012;
    sprintf(s,"%.*g", decimals, ((int)(pow(10, decimals)*(fabs(f) - abs((int)f)) +0.5))/pow(10,decimals));
    printf(" additional : %d%s\n", (int)f, s+1);    

    f = 123456.0123456;
    sprintf(s,"%.*g", decimals, ((int)(pow(10, decimals)*(fabs(f) - abs((int)f)) +0.5))/pow(10,decimals));
    printf(" additional : %d%s\n", (int)f, s+1);    
    f = 123456.0123456;
    printf(" test: %f\n", f);
    sprintf(s,"%.*g", decimals, ((int)(pow(10, decimals)*(fabs(f) - abs((int)f)) +0.5))/pow(10,decimals));
    printf(" additional : %d%s\n", (int)f, s+1);    

    f = 1.01;
    printf(">%.8f\n", f);
    f = 1.01;
    printf(">%.7f\n", f);

/*    f = 1234.56789;*/
    //printf("%d.%.0f\n", (int)f, 1000*(f-(int)f));
    //f = 1234.5;
    //printf("%d.%.0f\n", (int)f, 1000*(f-(int)f));

/*    f = 1234.5;*/
    //printf("%.0d%.8f\n", (int)f/10, f-((int)f-(int)f%10));
    //f = 1234.5134;
    //printf("%.0d%.8f\n", (int)f/10, f-((int)f-(int)f%10));
    //f = 1234.51234000323232;
    //printf("%.0d%.8f\n", (int)f/10, f-((int)f-(int)f%10));
    //f = 4.51234000323232;
    //printf("%.0d%.8f\n", (int)f/10, f-((int)f-(int)f%10));
    //f = 12345678911.51234000323232;
    //printf("%f\n", f);
    //printf("%.0d%.8f\n", (int)f/10, f-((int)f-(int)f%10));

    return 0;
}
