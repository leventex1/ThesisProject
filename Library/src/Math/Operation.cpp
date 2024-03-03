#include "Operation.h"
#include <assert.h>
#include <random>
#include <limits>

#include <MogiAccelerator.h>

#include <future>
#include <mutex>
#include "../ThreadPool.h"


namespace_start

Tensor2D SliceTensor(const Tensor3D& tensor, size_t depth)
{
	assert(depth < tensor.GetDepth() && "Depth is out of range.");

	return Tensor2D(tensor.GetRows(), tensor.GetCols(), tensor.GetData() + depth * tensor.GetRows() * tensor.GetCols());
}

Tensor2D CreateWatcher(Tensor2D& tensor, size_t row, size_t col, size_t rows, size_t cols, size_t skipRows, size_t skipCols)
{
	assert(row + (rows - 1) * (1 + skipRows) < tensor.GetRows() &&
		col + (cols - 1) * (1 + skipCols) < tensor.GetCols()
		&& "Out of range tensor params!");

	size_t otherOffsetRows = tensor.GetOffsetRows() != 0 ? tensor.GetOffsetRows() : tensor.GetCols();
	size_t otherOffsetCols = tensor.GetOffsetCols() != 0 ? tensor.GetOffsetCols() : 1;

	float* offsetData = tensor.GetData() + tensor.CalculateIndex(row, col);

	size_t offsetRows = otherOffsetRows * (1 + skipRows);
	size_t offsetCols = otherOffsetCols * (1 + skipCols);

	return Tensor2D(rows, cols, offsetRows, offsetCols, offsetData);
}

Tensor2D CreateWatcher(Tensor3D& tensor, size_t depth)
{
	assert(depth < tensor.GetDepth() && "Depth is out of range.");

	return Tensor2D(tensor.GetRows(), tensor.GetCols(), 0, 0, tensor.GetData() + depth * tensor.GetRows() * tensor.GetCols());
}

Tensor3D CreateWatcher(Tensor2D& tensor)
{
	return Tensor3D(tensor.GetRows(), tensor.GetCols(), 1, tensor.GetData());
}

Tensor3D CreateWatcher(Tensor3D& tensor, size_t fromDepth, size_t depth)
{
	assert(fromDepth + depth <= tensor.GetDepth() && "Depth is out of range.");

	return Tensor3D(tensor.GetRows(), tensor.GetCols(), depth, tensor.GetData() + fromDepth * tensor.GetRows() * tensor.GetCols());
}

Tensor2D Random2D(size_t rows, size_t cols, float min, float max)
{
	std::random_device rd;
	Tensor2D res(rows, cols, [&]() -> float {
		float r = (float)rd() / (float)rd.max();
		return min + r * (max - min);
	});
	return res;
}

Tensor3D Random3D(size_t rows, size_t cols, size_t depth, float min, float max)
{
	std::random_device rd;
	Tensor3D res(rows, cols, depth);
	res.Map([&](float v) -> float {
		float r = (float)rd() / (float)rd.max();
		return min + r * (max - min);
	});
	return res;
}

Tensor2D Map(const Tensor2D& t, std::function<float(float v)> mapper)
{
	Tensor2D res = t;
	for (size_t i = 0; i < res.GetRows(); i++)
		for (size_t j = 0; j < res.GetCols(); j++)
			res.SetAt(i, j, mapper(res.GetAt(i, j)));
	return res;
}

Tensor3D Map(const Tensor3D& t, std::function<float(float v)> mapper)
{
	Tensor3D res = t;
	for (size_t k = 0; k < res.GetDepth(); k++)
		for (size_t i = 0; i < res.GetRows(); i++)
			for (size_t j = 0; j < res.GetCols(); j++)
				res.SetAt(i, j, k, mapper(res.GetAt(i, j, k)));
	return res;
}

Tensor2D Add(const Tensor2D& t1, const Tensor2D& t2)
{
	Tensor2D res = t1;
	res.Add(t2);
	return res;
}

Tensor2D Sub(const Tensor2D& t1, const Tensor2D& t2)
{
	Tensor2D res = t1;
	res.Sub(t2);
	return res;
}

Tensor2D Mult(const Tensor2D& t1, const Tensor2D& t2)
{
	Tensor2D res = t1;
	res.Mult(t2);
	return res;
}

Tensor2D Div(const Tensor2D& t1, const Tensor2D& t2)
{
	Tensor2D res = t1;
	res.Div(t2);
	return res;
}

std::pair<size_t, size_t> MaxPos(const Tensor2D& tensor)
{
	assert(tensor.GetSize() > 0 && "No value in tensor!");

	std::pair<size_t, size_t> res = { 0, 0 };
	float min = tensor.GetAt(0, 0);

	for (size_t i = 0; i < tensor.GetRows(); i++)
	{
		for (size_t j = 0; j < tensor.GetCols(); j++)
		{
			if (tensor.GetAt(i, j) > min)
			{
				min = tensor.GetAt(i, j);
				res = { i, j };
			}
		}
	}

	return res;
}

Tensor2D Transpose(const Tensor2D& t)
{
	Tensor2D res(t.GetCols(), t.GetRows());

	for (size_t i = 0; i < res.GetRows(); i++)
	{
		for (size_t j = 0; j < res.GetCols(); j++)
		{
			res.SetAt(i, j, t.GetAt(j, i));
		}
	}

	return res;
}

void AsyncMatrixMult(int startRow, int endRow, int startCol, int endCol, Tensor2D* res, const Tensor2D* left, const Tensor2D* right)
{
	for (int row = startRow; row < endRow; ++row) {
		for (int col = startCol; col < endCol; ++col) {

			float product = 0.0f;
			for (size_t t = 0; t < left->GetCols(); t++)
			{
				product += left->GetAt(row, t) * right->GetAt(t, col);
			}
			res->SetAt(row, col, product);
		}
	}
}

Tensor2D MatrixMult(const Tensor2D& left, const Tensor2D& right, bool useCuda)
{
	if (useCuda)
	{
		return accelerator::MatrixMultCUDA(left, right);
	}

	assert(left.GetCols() == right.GetRows() && "Matrix params for matrix multiplication not math! Left.column != Right.rows");
	Tensor2D res(left.GetRows(), right.GetCols());

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int rowsPerThread = res.GetRows() / numThreads;

	std::vector<std::future<void>> threads;
	for (size_t i = 0; i < numThreads; i++)
	{
		size_t startRow = i * rowsPerThread;
		int endRow = (i == numThreads - 1) ? res.GetRows() : (i + 1) * rowsPerThread;

		threads.emplace_back(
			pool->enqueue(AsyncMatrixMult, startRow, endRow, 0, res.GetCols(), &res, &left, &right)
		);
	}

	for (auto& thread : threads) {
		thread.get();
	}
#else
	for (size_t row = 0; row < res.GetRows(); row++)
	{
		for (size_t col = 0; col < res.GetCols(); col++)
		{
			float product = 0.0f;
			for (size_t t = 0; t < left.GetCols(); t++)
			{
				product += left.GetAt(row, t) * right.GetAt(t, col);
			}
			res.SetAt(row, col, product);
		}
	}
#endif // ASYNC

	return res;
}

void AsyncMatrixMultLeftTranspose(int startRow, int endRow, int startCol, int endCol, Tensor2D* res, const Tensor2D* left, const Tensor2D* right)
{
	for (int row = startRow; row < endRow; ++row) {
		for (int col = startCol; col < endCol; ++col) {

			float product = 0.0f;
			for (size_t t = 0; t < left->GetRows(); t++)
			{
				product += left->GetAt(t, row) * right->GetAt(t, col);
			}
			res->SetAt(row, col, product);
		}
	}
}

Tensor2D MatrixMultLeftTranspose(const Tensor2D& left, const Tensor2D& right)
{
	assert(left.GetRows() == right.GetRows() && "Matrix params for matrix multiplication not math! Left.rows(after taking transpose) != Right.rows");
	Tensor2D res(left.GetCols(), right.GetCols());

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int rowsPerThread = res.GetRows() / numThreads;

	std::vector<std::future<void>> threads;
	for (size_t i = 0; i < numThreads; i++)
	{
		size_t startRow = i * rowsPerThread;
		int endRow = (i == numThreads - 1) ? res.GetRows() : (i + 1) * rowsPerThread;

		threads.emplace_back(
			pool->enqueue(AsyncMatrixMultLeftTranspose, startRow, endRow, 0, res.GetCols(), &res, &left, &right)
		);
	}

	for (auto& thread : threads) {
		thread.get();
	}
#else
	for (size_t row = 0; row < res.GetRows(); row++)
	{
		for (size_t col = 0; col < res.GetCols(); col++)
		{
			float product = 0.0f;
			for (size_t t = 0; t < left.GetRows(); t++)
			{
				product += left.GetAt(t, row) * right.GetAt(t, col);
			}
			res.SetAt(row, col, product);
		}
	}
#endif

	return res;
}

void AsyncMatrixMulRightTranspose(int startRow, int endRow, int startCol, int endCol, Tensor2D* res, const Tensor2D* left, const Tensor2D* right)
{
	for (int row = startRow; row < endRow; ++row) {
		for (int col = startCol; col < endCol; ++col) {

			float product = 0.0f;
			for (size_t t = 0; t < left->GetCols(); t++)
			{
				product += left->GetAt(row, t) * right->GetAt(col, t);
			}
			res->SetAt(row, col, product);
		}
	}
}

Tensor2D MatrixMultRightTranspose(const Tensor2D& left, const Tensor2D& right)
{
	assert(left.GetCols() == right.GetCols() && "Matrix params for matrix multiplication not math! Left.rows(after taking transpose) != Right.rows");
	Tensor2D res(left.GetRows(), right.GetRows());

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int rowsPerThread = res.GetRows() / numThreads;

	std::vector<std::future<void>> threads;
	for (size_t i = 0; i < numThreads; i++)
	{
		size_t startRow = i * rowsPerThread;
		int endRow = (i == numThreads - 1) ? res.GetRows() : (i + 1) * rowsPerThread;

		threads.emplace_back(
			pool->enqueue(AsyncMatrixMulRightTranspose, startRow, endRow, 0, res.GetCols(), &res, &left, &right)
		);
	}

	for (auto& thread : threads) {
		thread.get();
	}
#else
	for (size_t row = 0; row < res.GetRows(); row++)
	{
		for (size_t col = 0; col < res.GetCols(); col++)
		{
			float product = 0.0f;
			for (size_t t = 0; t < left.GetCols(); t++)
			{
				product += left.GetAt(row, t) * right.GetAt(col, t);
			}
			res.SetAt(row, col, product);
		}
	}
#endif

	return res;
}

size_t CalcConvSize(size_t inputSize, size_t kernelSize, size_t stride, size_t padding)
{
	return (inputSize - kernelSize + 2 * padding) / stride + 1;
}

float KernelOperation(const Tensor2D& window, const Tensor2D& kernel)
{
	assert(window.GetRows() == kernel.GetRows() && window.GetCols() == kernel.GetCols() && "Window and kernel params not match!");
	
	float res = 0.0f;
	for (size_t i = 0; i < window.GetSize(); i++)
	{
		size_t windowIndex = window.TraverseTo(i);
		size_t kernelIndex = kernel.TraverseTo(i);
		res += window.GetData()[windowIndex] * kernel.GetData()[kernelIndex];
	}
	return res;
}

void Convolution(Tensor2D& output, const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor!");

	for (size_t y = 0; y < outputRows; y++)
	{
		for (size_t x = 0; x < outputCols; x++)
		{
			float sum = 0.0f;
			for (size_t ky = 0; ky < kernel.GetRows(); ky++)
			{
				for (size_t kx = 0; kx < kernel.GetCols(); kx++)
				{
					int posY = y * stride + ky - padding;
					int posX = x * stride + kx - padding;

					if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
						sum += input.GetAt(posY, posX) * kernel.GetAt(ky, kx);
					}
				}
			}
			output.SetAt(y, x, output.GetAt(y, x) + sum);
		}
	}
}

void ConvolutionKernelFlip(Tensor2D& output, const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor!");

	for (size_t y = 0; y < outputRows; y++)
	{
		for (size_t x = 0; x < outputCols; x++)
		{
			float sum = 0.0f;
			for (size_t ky = 0; ky < kernel.GetRows(); ky++)
			{
				for (size_t kx = 0; kx < kernel.GetCols(); kx++)
				{
					int posY = y * stride + ky - padding;
					int posX = x * stride + kx - padding;

					size_t flippedKy = kernel.GetRows() - 1 - ky;
					size_t flippedKx = kernel.GetCols() - 1 - kx;

					if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
						sum += input.GetAt(posY, posX) * kernel.GetAt(flippedKy, flippedKx);
					}
				}
			}
			output.SetAt(y, x, output.GetAt(y, x) + sum);
		}
	}
}

void Convolution(Tensor2D& output, const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(input.GetDepth() == kernel.GetDepth() && "Input and kernel depth not match!");
	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor!");

	for (size_t y = 0; y < outputRows; y++)
	{
		for (size_t x = 0; x < outputCols; x++)
		{
			float sum = 0.0f;
			for (size_t ky = 0; ky < kernel.GetRows(); ky++)
			{
				for (size_t kx = 0; kx < kernel.GetCols(); kx++)
				{
					for (size_t d = 0; d < kernel.GetDepth(); d++)
					{
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX, d) * kernel.GetAt(ky, kx, d);
						}
					}

				}
			}
			output.SetAt(y, x, output.GetAt(y, x) + sum);
		}
	}
}

void ConvolutionKernelFlip(Tensor2D& output, const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(input.GetDepth() == kernel.GetDepth() && "Input and kernel depth not match!");
	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor!");

	for (size_t y = 0; y < outputRows; y++)
	{
		for (size_t x = 0; x < outputCols; x++)
		{
			float sum = 0.0f;
			for (size_t ky = 0; ky < kernel.GetRows(); ky++)
			{
				for (size_t kx = 0; kx < kernel.GetCols(); kx++)
				{
					for (size_t d = 0; d < kernel.GetDepth(); d++)
					{
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						size_t flippedKy = kernel.GetRows() - 1 - ky;
						size_t flippedKx = kernel.GetCols() - 1 - kx;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX, d) * kernel.GetAt(flippedKy, flippedKx, d);
						}
					}

				}
			}
			output.SetAt(y, x, output.GetAt(y, x) + sum);
		}
	}
}

void Convolution(Tensor3D& output, const Tensor3D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor size!");
	assert(output.GetDepth() == input.GetDepth() && "Invalid output tensor depth!");


	for (size_t d = 0; d < output.GetDepth(); d++)
	{
		for (size_t y = 0; y < outputRows; y++)
		{
			for (size_t x = 0; x < outputCols; x++)
			{
				float sum = 0.0f;
				for (size_t ky = 0; ky < kernel.GetRows(); ky++)
				{
					for (size_t kx = 0; kx < kernel.GetCols(); kx++)
					{
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX, d) * kernel.GetAt(ky, kx);
						}

					}
				}
				output.SetAt(y, x, d, output.GetAt(y, x, d) + sum);
			}
		}
	}
}

void ConvolutionKernelFlip(Tensor3D& output, const Tensor3D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor size!");
	assert(output.GetDepth() == input.GetDepth() && "Invalid output tensor depth!");


	for (size_t d = 0; d < output.GetDepth(); d++)
	{
		for (size_t y = 0; y < outputRows; y++)
		{
			for (size_t x = 0; x < outputCols; x++)
			{
				float sum = 0.0f;
				for (size_t ky = 0; ky < kernel.GetRows(); ky++)
				{
					for (size_t kx = 0; kx < kernel.GetCols(); kx++)
					{
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						size_t flippedKy = kernel.GetRows() - 1 - ky;
						size_t flippedKx = kernel.GetCols() - 1 - kx;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX, d) * kernel.GetAt(flippedKy, flippedKx);
						}

					}
				}
				output.SetAt(y, x, d, output.GetAt(y, x, d) + sum);
			}
		}
	}
}

void ConvolutionKernelFlip(Tensor3D& output, const Tensor2D& input, const Tensor3D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor size!");
	assert(output.GetDepth() == kernel.GetDepth() && "Invalid output tensor depth!");

	for (size_t d = 0; d < output.GetDepth(); d++)
	{
		for (size_t y = 0; y < outputRows; y++)
		{
			for (size_t x = 0; x < outputCols; x++)
			{
				float sum = 0.0f;
				for (size_t ky = 0; ky < kernel.GetRows(); ky++)
				{
					for (size_t kx = 0; kx < kernel.GetCols(); kx++)
					{
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						size_t flippedKy = kernel.GetRows() - 1 - ky;
						size_t flippedKx = kernel.GetCols() - 1 - kx;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX) * kernel.GetAt(flippedKy, flippedKx, d);
						}

					}
				}
				output.SetAt(y, x, d, output.GetAt(y, x, d) + sum);
			}
		}
	}
}


Tensor2D Convolution(const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	Tensor2D output(outputRows, outputCols);

	Convolution(output, input, kernel, stride, padding);

	return output;
}

Tensor2D ConvolutionKernelFlip(const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	Tensor2D output(outputRows, outputCols);

	ConvolutionKernelFlip(output, input, kernel, stride, padding);

	return output;
}

Tensor2D Convolution(const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	Tensor2D output(outputRows, outputCols);

	Convolution(output, input, kernel, stride, padding);

	return output;
}

Tensor2D ConvolutionKernelFlip(const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	Tensor2D output(outputRows, outputCols);

	ConvolutionKernelFlip(output, input, kernel, stride, padding);

	return output;
}

namespace_end