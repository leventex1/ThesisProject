#pragma once
#include <functional>
#include "Core.h"
#include "Tensor2D.h"
#include "Tensor3D.h"


namespace_start

LIBRARY_API Tensor2D SliceTensor(const Tensor3D& tensor, size_t depth);
LIBRARY_API Tensor2D CreateWatcher(Tensor2D& tensor, size_t row, size_t col, size_t rows, size_t cols, size_t skipRows, size_t skipCols);
LIBRARY_API Tensor2D CreateWatcher(Tensor3D& tensor, size_t depth);
LIBRARY_API Tensor3D CreateWatcher(Tensor2D& tensor);

LIBRARY_API Tensor2D Random2D(size_t rows, size_t cols, float min, float max);
LIBRARY_API Tensor3D Random3D(size_t rows, size_t cols, size_t depth, float min, float max);

LIBRARY_API Tensor2D Map(const Tensor2D& t, std::function<float(float v)> mapper);
LIBRARY_API Tensor3D Map(const Tensor3D& t, std::function<float(float v)> mapper);

LIBRARY_API Tensor2D Add(const Tensor2D& t1, const Tensor2D& t2);
LIBRARY_API Tensor2D Sub(const Tensor2D& t1, const Tensor2D& t2);
LIBRARY_API Tensor2D Mult(const Tensor2D& t1, const Tensor2D& t2);
LIBRARY_API Tensor2D Div(const Tensor2D& t1, const Tensor2D& t2);

LIBRARY_API Tensor2D Transpose(const Tensor2D& t);

LIBRARY_API Tensor2D MatrixMult(const Tensor2D& left, const Tensor2D& right);
// Calculates the matrix multiplication, but take the left matrix as a transpose matrix without extra calculation.
LIBRARY_API Tensor2D MatrixMultLeftTranspose(const Tensor2D& left, const Tensor2D& right);
// Calculates the matrix multiplication, but take the right matrix as a transpose matrix without extra calculation.
LIBRARY_API Tensor2D MatrixMultRightTranspose(const Tensor2D& left, const Tensor2D& right);

LIBRARY_API size_t CalcConvSize(size_t inputSize, size_t kernelSize, size_t stride, size_t padding);
LIBRARY_API float KernelOperation(const Tensor2D& window, const Tensor2D& kernel);

// Preform a convolutional operation on input with kernel with a zero padding and overwrites output.
LIBRARY_API void Convolution(Tensor2D& output, const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding);
// Preform a convolutional operation on input with kernel with a zero padding.
LIBRARY_API Tensor2D Convolution(const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding);

namespace_end
