#ifndef LAPACKE_BACKEND_H
#define LAPACKE_BACKEND_H

#include "lapacke.h"

int solveSystemOfLinearEquations(double *a, double *b, int n, double *res)
{
    int ldb = 1;
    int lda = n;
    int nrhs = 1;
    int *ipiv = malloc(sizeof(int) * n);
    int i = 0;
    for(i = 0; i < n; ++i)
        res[i] = b[i];
    int info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, res, ldb );
    free(ipiv);
    return info;
}


#endif
