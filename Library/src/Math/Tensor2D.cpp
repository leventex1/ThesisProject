#include "Tensor2D.h"
#include <assert.h>


namespace_start

Tensor2D::Tensor2D() : m_Rows(0), m_Cols(0), Tensor() { }

Tensor2D::Tensor2D(size_t rows, size_t cols, float value)
	: m_Rows(rows), m_Cols(cols), Tensor(rows * cols, value)
{
}

Tensor2D::Tensor2D(const std::initializer_list<std::initializer_list<float>>& initList)
	: Tensor()
{
	m_Rows = initList.size();
	m_Cols = initList.begin()->size();

	Alloc(m_Rows * m_Cols);

	size_t index = 0;
	for (auto i = initList.begin(); i != initList.end(); i++)
	{
		assert(i->size() == m_Cols && "Tensor2D missmatched params numbers!");
		for (auto j = i->begin(); j != i->end(); j++)
		{
			Tensor::SetAt(index, (*j));
			index++;
		}
	}
}

Tensor2D::Tensor2D(size_t rows, size_t cols, float* data)
	: m_Rows(rows), m_Cols(cols), Tensor(data)
{
}

Tensor2D::Tensor2D(size_t rows, size_t cols, const float* data)
	: m_Rows(rows), m_Cols(cols), Tensor(data, rows * cols)
{
}

float Tensor2D::GetAt(size_t row, size_t col) const
{
	size_t index = row * m_Cols + col;
	return Tensor::GetAt(index);
}

void Tensor2D::SetAt(size_t row, size_t col, float value)
{
	size_t index = row * m_Cols + col;
	Tensor::SetAt(index, value);
}

namespace_end