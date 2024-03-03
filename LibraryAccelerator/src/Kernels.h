#include "cuda_runtime.h"

__global__ void MatrixMultKernel(const float* A, const float* B, float* C, int aRows, int aCols, int bCols);
__global__ void MatrixMultRightTranposeKernel(const float* A, const float* B, float* C, int aRows, int aCols, int bCols);
__global__ void MatrixMultLeftTranposeKernel(const float* A, const float* B, float* C, int ARows, int ACols, int BCols);
