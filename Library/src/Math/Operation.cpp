#include "Operation.h"
#include <assert.h>
#include <random>


namespace_start

Tensor2D SliceTensor(const Tensor3D& tensor, size_t depth)
{
	assert(depth < tensor.GetDepth() && "Depth is out of range.");

	return Tensor2D(tensor.GetRows(), tensor.GetCols(), tensor.GetData() + depth * tensor.GetRows() * tensor.GetCols());
}

Tensor2D CreateWatcher(Tensor3D& tensor, size_t depth)
{
	assert(depth < tensor.GetDepth() && "Depth is out of range.");

	return Tensor2D(tensor.GetRows(), tensor.GetCols(), tensor.GetData() + depth * tensor.GetRows() * tensor.GetCols());
}

Tensor3D CreateWatcher(Tensor2D& tensor)
{
	return Tensor3D(tensor.GetRows(), tensor.GetCols(), 1, tensor.GetData());
}

Tensor2D Random2D(size_t rows, size_t cols, float min, float max)
{
	std::random_device rd;
	Tensor2D res(rows, cols);
	res.Map([&](float v) -> float {
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
	res.Map(mapper);
	return res;
}

Tensor3D Map(const Tensor3D& t, std::function<float(float v)> mapper)
{
	Tensor3D res = t;
	res.Map(mapper);
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

Tensor2D MatrixMult(const Tensor2D& left, const Tensor2D& right)
{
	assert(left.GetCols() == right.GetRows() && "Matrix params for matrix multiplication not math! Left.column != Right.rows");
	Tensor2D res(left.GetRows(), right.GetCols());

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

	return res;
}

Tensor2D MatrixMultLeftTranspose(const Tensor2D& left, const Tensor2D& right)
{
	assert(left.GetRows() == right.GetRows() && "Matrix params for matrix multiplication not math! Left.rows(after taking transpose) != Right.rows");
	Tensor2D res(left.GetCols(), right.GetCols());

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

	return res;
}

Tensor2D MatrixMultRightTranspose(const Tensor2D& left, const Tensor2D& right)
{
	assert(left.GetCols() == right.GetCols() && "Matrix params for matrix multiplication not math! Left.rows(after taking transpose) != Right.rows");
	Tensor2D res(left.GetRows(), right.GetRows());

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

	return res;
}


namespace_end