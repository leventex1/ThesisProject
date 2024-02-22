#pragma once
#include <functional>
#include "Core.h"
#include "Tensor2D.h"


namespace_start

LIBRARY_API Tensor2D Random2D(size_t rows, size_t cols, float min, float max);

LIBRARY_API Tensor2D Map(const Tensor2D& t, std::function<float(float v)> mapper);

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

namespace_end
