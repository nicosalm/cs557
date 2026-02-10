#include "Reductions.h"

#include <algorithm>

float Norm(const float (&x)[XDIM][YDIM][ZDIM])
{
    float result = 0.;

// max is associative and commutative and thus a reduction
#pragma omp parallel for reduction(max:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        result = std::max(result, std::abs(x[i][j][k]));

    return result;
}

float InnerProduct(const float (&x)[XDIM][YDIM][ZDIM], const float (&y)[XDIM][YDIM][ZDIM])
{
    double result = 0.;

// ... and so is addition
#pragma omp parallel for reduction(+:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        result += (double) x[i][j][k] * (double) y[i][j][k];    // we cast as a double because floats cannot represent numbers super high:
                                                                // example? run below & see what happens lol:
                                                                // #include <stdio.h>
                                                                // #include <limits.h>
                                                                //
                                                                // int main(void) {
                                                                //    float x = 0.0;
                                                                //    for (int i = 0; i < INT_MAX; i++) {
                                                                //        x += 1.0;
                                                                //        printf("%f\n", x);
                                                                //    }
                                                                //}

    return (float) result;
}
