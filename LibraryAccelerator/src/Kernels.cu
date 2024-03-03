#include "Kernels.h"

#include "device_launch_parameters.h"


__global__ void MatrixMultKernel(const float* A, const float* B, float* C, int ARows, int ACols, int BCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ARows && col < BCols) {
        float sum = 0.0;
        for (int i = 0; i < BCols; ++i) {
            sum += A[row * BCols + i] * B[i * BCols + col];
        }
        C[row * BCols + col] = sum;
    }
}

__global__ void MatrixMultRightTranposeKernel(const float* A, const float* B, float* C, int ARows, int ACols, int BRows)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ARows && col < BRows) {
        float sum = 0.0f;
        for (int e = 0; e < ACols; ++e) {
            sum += A[row * ACols + e] * B[col * ACols + e]; // Accessing B as if it's transposed
        }
        C[row * BRows + col] = sum;
    }
}

__global__ void MatrixMultLeftTranposeKernel(const float* A, const float* B, float* C, int ARows, int ACols, int BCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ACols && col < BCols) { // A is transposed, use ACols for row checks
        float sum = 0.0f;
        for (int e = 0; e < ARows; ++e) { // ARows is used here, reflecting the transposed dimension
            sum += A[e * ACols + row] * B[e * BCols + col]; // Access A as transposed
        }
        C[row * BCols + col] = sum;
    }
}