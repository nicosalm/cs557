#pragma once

#include <vector>
#include <map>
#include <stdexcept>

#include "CSRMatrix.h"

/**
 * This is very slow but convenient
 */
struct CSRMatrixHelper
{
    std::vector<std::map<int,float> > mSparseRows;

    CSRMatrixHelper(const int size) { mSparseRows.resize(size); }

    float& operator() (const int i, const int j)
    {
        if (i < 0 || i >= mSparseRows.size() || j < 0 || j >= mSparseRows.size())
            throw std::logic_error("Matrix index out of bounds");
        return mSparseRows[i].insert( {j, 0.} ).first->second;
    }

    CSRMatrix ConvertToCSRMatrix()
    {
        int N = mSparseRows.size(); // Size of matrix
        int NNZ = 0; // Number of non-zero entries
        for (int i = 0; i < N; i++) NNZ += mSparseRows[i].size();

        CSRMatrix matrix { N }; // Initialize just matrix.mSize
        matrix.mRowOffsets.reset(new int [N + 1]); // Need a sentinel value in the end
        matrix.mColumnIndices.reset(new int [NNZ]);
        matrix.mValues.reset(new float [NNZ]);

        auto rowOffsets = matrix.GetRowOffsets(); // where does the 0..Nth row begin? ..., i=k, i+1=k+3, ...
                                                  // offsets[i] = how many elements are there in the matrix in rows < i
                                                  // where does the ith row begin? -> offsets[i] (caveat: if it exists)
                                                  // where does the ith row end? -> offsets[i+1] - 1
                                                  // See: MatVecMultiply.cpp
        auto columnIndices = matrix.GetColumnIndices(); // ... 2, 7, 9, ...
        auto values = matrix.GetValues(); // ..., A_{i2}, A_{i7}, A_{i9}, ...

        // auto keyword (new C++ 11) -- you do not need to say specifically float*,
        // because we know GetRowOffsets returns a float* (wow)

        rowOffsets[0] = 0;
        for (int i = 0, k = 0; i < N; i++)
        {
            rowOffsets[i + 1] = rowOffsets[i] + mSparseRows[i].size(); // Mark where this row ends
            for (auto it = mSparseRows[i].begin(); it != mSparseRows[i].end(); it++)
            {
                columnIndices[k] = it->first;
                values[k] = it->second;
                k++;
            }
        }

        return matrix;
    }
};
