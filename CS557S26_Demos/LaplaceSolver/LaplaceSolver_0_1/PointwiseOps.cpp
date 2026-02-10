#include "PointwiseOps.h"

/**
 * Reduction: A context-specific name for an associative operation; we take a bunch of numbers and digest them into a result. We could break the operations (which are)
 * the same on all the numbers, into partial reductions (and in theory with unlimited precision) the result would be the same.
 * e.g., Max { Max { a, b }, c } = Max { a, Max { c, b }}
 *
 * The benefit for parallel computing is we can distribute this across many processors since everything is ultimately being added to the same accumulator.
 * Works for associative & commutative operators, but not things like Substraction/Division
 *
 * A special treatment of the reduction operation is needed to avoid false sharing.
 *
 * "+=" reads whatever value is already in that var and adds, then stores result back. If two cores try to do this op without any coordination, they may have read the
 * same original value, individually added their own contributions, and then try to commit back result. The final result is one of the two values, but it won't be the right
 * result.
 *
 * Could say, "stop all cores from doing ops except the one that is doing the op" e.g., monolithic/atomic ops -- terrible for performance. Instead, allow each core to
 * compute partial sum on its own. Then, collect all results.
 *
 * If any cache line:
 * Given a read request: has any other cache obtained a writeable copy (exclusive copy)?
 *      If yes: invalidate all.
 *      If no: acquire another non-exclusive copy
 * Given a write request: invalidate all copies. then, aquire an exclusive access line.
 *
 * Requires access and time; if two cores are playing tug of war to aquire write access to a single cache line, every other access will have to pay the cost of
 * expunging everything, pulling in to main memory, etc. --> Happens when two cores are trying to access values right next to one another.
 *
 * E.g.:
 *
 * float partial_sum[20]
 * pragma omp parallel for:
 *      for i = 0...N:
 *          partial_sum[my_core_id()] += a[i]
 *
 * 0...M:
 *      Give 0...100K -> Core 0
 *      Give 100K ... 1M -> Core 1
 *
 * Core 0 tries to partial_sum[0] += a[0]
 * Core 1 tries to partial_sum[1] += a[100K]
 *
 * Suppose Core 0 already has the entire cache line in its cache; it goes ahead and reads in a[0].
 * Simultaneously, Core 1 wants to read in a[1]. No non-exclusive copy available (Core 0 has an exlusive copy).
 * Expunge the cache line. Wait for Core 1's exclusive access to come back.
 * Core 0 finishes computing the addition but its cache line is gone! Where did it go!
 *
 * Best way to avoid false sharing: every different core keeps a completely separate/isolated variable where it deposits its local result e.g., a local var allocated on the stack
 * Or, if you have to allocate an array, pad it enough so that you use only 1 of every 16 elements etc.
 *
 * /OpenMP does this automatically!/ with the `reduction` keyword. It creates separate local vars, then reduces after to add them all together.
 *
 */
void Copy(const float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM])
{
#pragma omp parallel for
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        y[i][j][k] = x[i][j][k];
}

void Saxpy(const float (&x)[XDIM][YDIM][ZDIM], const float (&y)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM],
    const float scale)
{
    // Should we use OpenMP parallel for here?
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        z[i][j][k] = x[i][j][k] * scale + y[i][j][k];
}

/*
 * Suppose by contrast,
 * K = # nonzero entries
 * N = # rows/cols
 *
 * float x[4], y[4] # y <- A*x
 * for i = 0...K-1  // why not add a pragma omp for here? -- A: no strong guarantee that cores won't touch
 *      y[row[k]] = values[k] * x[col[k]]
 *
 * We use CSR (Compressed Sparse Row) which uses the fact that the rows array (if sorted) can be compressed.
 * Invariant for CSR? (TODO)
 */
