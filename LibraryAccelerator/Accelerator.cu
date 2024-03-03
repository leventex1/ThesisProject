#include "MogiAccelerator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "src/Kernels.h"


namespace_accelerator_start

float* ToDevicePtr(const Tensor2D& tensor, bool copy=true)
{
    float* dPtr;
    size_t size = tensor.GetSize() * sizeof(float);

    cudaMalloc(&dPtr, size);
    if (copy)
    {
        cudaMemcpy(dPtr, tensor.GetData(), size, cudaMemcpyHostToDevice);
    }

    return dPtr;
}

void CopyToHost(Tensor2D& dest, float* deviceSrource)
{
    cudaMemcpy(dest.GetData(), deviceSrource, dest.GetSize() * sizeof(float), cudaMemcpyDeviceToHost);
}

Tensor2D MatrixMultCUDA(const Tensor2D& left, const Tensor2D& right)
{
    if (left.GetCols() != right.GetRows())
    {
        throw -1;
    }

    Tensor2D res(left.GetRows(), right.GetCols());

    float* dA = ToDevicePtr(left);
    float* dB = ToDevicePtr(right);
    float* dC = ToDevicePtr(res, false);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((right.GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (left.GetRows()  + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatrixMultKernel<< <blocksPerGrid, threadsPerBlock >> > (dA, dB, dC, left.GetRows(), left.GetCols(), right.GetCols());

    CopyToHost(res, dC);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return res;
}

Tensor2D MatrixMultRightTransposeCUDA(const Tensor2D& left, const Tensor2D& right)
{
    if (left.GetCols() != right.GetCols())
    {
        throw -1;
    }

    Tensor2D res(left.GetRows(), right.GetRows());

    float* dA = ToDevicePtr(left);
    float* dB = ToDevicePtr(right);
    float* dC = ToDevicePtr(res, false);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((right.GetRows() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (left.GetRows() + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatrixMultRightTranposeKernel << <blocksPerGrid, threadsPerBlock >> > (dA, dB, dC, left.GetRows(), left.GetCols(), right.GetRows());

    CopyToHost(res, dC);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return res;
}

Tensor2D MatrixMultLeftTransposeCUDA(const Tensor2D& left, const Tensor2D& right)
{
    if (left.GetRows() != right.GetRows())
    {
        throw - 1;
    }

    Tensor2D res(left.GetCols(), right.GetCols());

    float* dA = ToDevicePtr(left);
    float* dB = ToDevicePtr(right);
    float* dC = ToDevicePtr(res, false);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((right.GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (left.GetCols() + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatrixMultLeftTranposeKernel<< <blocksPerGrid, threadsPerBlock >> > (dA, dB, dC, left.GetRows(), left.GetCols(), right.GetCols());

    CopyToHost(res, dC);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return res;
}

namespace_accelerator_end