#include <stdio.h>
#include "lapacke.h"

#include "hi.h"

void hi(char* name) {
    printf("awesome %s\n", name);
}

void eigen()
{
    double evalues[4];
    double evectors[4*4] = {
        1, 2, 3, 4, 
        2, 6, 7, 8, 
        3, 7, 11, 15, 
        4, 8, 15, 16 
    };

    /*for(int row = 0; row < matrixRows; ++row)*/
        /*for(int col = 0; col < matrixCols; ++col)*/
            /*evectors[row*matrixCols + col] = inputMatrix(row, col);*/

    int n = 4;
    int lda = 4;
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, evectors, lda, evalues);

    printf("awesome %f\n", evalues[0]);
    printf("awesome %f\n", evalues[1]);
    printf("awesome %f\n", evalues[2]);
    printf("awesome %f\n", evalues[3]);
}

void addTenToSeq(double *input, int size, double *output)
{
    int i;
    for(i = 0; i < size; ++i)
        output[i] = input[i] + 10;
}
