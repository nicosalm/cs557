#include "MatVecMultiply.h"

void MatVecMultiply(CSRMatrix& mat, const float *x, float *y)
{
    int N = mat.mSize;
    const auto rowOffsets = mat.GetRowOffsets();
    const auto columnIndices = mat.GetColumnIndices();
    const auto values = mat.GetValues();

/**
 *
 */
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        y[i] = 0.;
        for (int k = rowOffsets[i]; k < rowOffsets[i+1]; k++) {
            const int j = columnIndices[k];
            y[i] += values[k] * x[j];   // we do not know ahead of time what x[j] is -- this is
                                        // where you might have bad caching because there is nothing
                                        // stopping you from having completely random j's
                                        // but with LaplacianMatrix/stencil format, this isn't really a problem
                                        //
                                        // still need to shoulder cost of reading in all column and row indices, read x, write y:
                                        // n access for x, n access for x (some are hopefully cached), 7n accesses for values, 7n access for index
        }
    }
}
