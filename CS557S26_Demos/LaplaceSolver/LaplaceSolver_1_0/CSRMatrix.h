#pragma once

#include <memory>

/**
 * This interface is not optimized for individual quick inserts and removals;
 * cannot introduce non-existing element easily. See: CSRMatrixHelper.h to make matrices easily
 */
struct CSRMatrix
{
    int mSize;
    std::unique_ptr<int> mRowOffsets;   // std::unique_ptr -- wrapper around pointer of underlying type
                                        // when std::unique_ptr goes out of scope, if anything has been allocated to it,
                                        // that stuff is deallocated limitations on when you can assign a unique_ptr to
                                        // another unique_ptr effectively, guarding raw pointers
    std::unique_ptr<int> mColumnIndices;
    std::unique_ptr<float> mValues;

    int* GetRowOffsets() { return mRowOffsets.get(); }
    int* GetColumnIndices() { return mColumnIndices.get(); }
    float* GetValues() { return mValues.get(); }
};
