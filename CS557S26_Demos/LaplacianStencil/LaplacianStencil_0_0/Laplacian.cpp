#include "Laplacian.h"

void ComputeLaplacian(const float (&u)[XDIM][YDIM], float (&Lu)[XDIM][YDIM])
{

/* (openmp) spread work in some yet-to-be-determined way over the cores;
 * it can begin in one way, (e.g., 400/800 entries processed), but rebalance later.
 * intent:  take pieces of work identified by the loop which follows and distribute the individual iterations of the loop
 *          over different cores
 */
#pragma omp parallel for
    // for array in range, compute exactly that stencil application code from earlier
    for (int i = 1; i < XDIM-1; i++)    // note: cautious about bounds (not starting from 0 to last idx of every array e.g., XDIM-1)
    for (int j = 1; j < YDIM-1; j++)    // two gigantic images; trying to get stencil img to be correct for all but the boundary
        Lu[i][j] =
            -4 * u[i][j]    // at minimum cost: read entry u[i][j], write Lu[i][j]
            + u[i+1][j]     // two arrays 16K x 16K floats = 2GB total
            + u[i-1][j]     // At 80-90GB/s: 22-25ms (super optimistic, best case scenario)
            + u[i][j+1]
            + u[i][j-1];    // the efficiency of this execution is very high (not likely for workloads to exceed 80-90% of peak efficiency)
}
                            // this example is clearly memory-bound, 4 additions, 1 multiplication
                            // indices math hidden by compiler using pointer arithmetic

/*
 * things that can go wrong (2 biggies):
 * - mem bandwidth
 * - compute bandwidth
 * note:    you can do 100x more computations than you have memory to run, 100 ops per mem read/write
 *
 * mem reads held back by caches, prefetching, data organization
 * -> STREAM benchmark for mem bandwidth capabilities: https://www.cs.virginia.edu/stream/ref.html
 *      4 tests:    (1) copy arr a[] to [b] - "Copy"
 *                  (2) scale array a[] by a constant value - "Scale"
 *                  (3) add respective entries in a[] & b[] - "Add"
 *                  (4) multiply entries in a[] & b[] and add to c[] - "Triad"
 *
 *                  on benchmark computer, practical bandwidth = 80/90GB/s out of 138GB/s max
 */
