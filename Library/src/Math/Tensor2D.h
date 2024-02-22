#pragma once
#include "Core.h"
#include "Tensor.h"


namespace_start

class LIBRARY_API Tensor2D : public Tensor
{
public:
	Tensor2D();
	Tensor2D(size_t rows, size_t cols, float value = 0.0f);
	Tensor2D(const std::initializer_list<std::initializer_list<float>>& initList);

	inline virtual size_t GetSize() const { return m_Rows * m_Cols; }
	
	float GetAt(size_t row, size_t col) const;
	void SetAt(size_t row, size_t col, float value);

	inline size_t GetRows() const { return m_Rows; }
	inline size_t GetCols() const { return m_Cols; }

private:
	size_t m_Rows, m_Cols;
};

namespace_end
