#pragma once

#include "src/AcceleratorCore.h"

#include <Mogi.h>


namespace_accelerator_start

LIBRARY_API Tensor2D MatrixMultCUDA(const Tensor2D& left, const Tensor2D& right);
LIBRARY_API Tensor2D MatrixMultRightTransposeCUDA(const Tensor2D& left, const Tensor2D& right);
LIBRARY_API Tensor2D MatrixMultLeftTransposeCUDA(const Tensor2D& left, const Tensor2D& right);

namespace_accelerator_end