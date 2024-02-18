#pragma once
#include "Core.h"
#include <functional>
#include <string>


namespace_start

class LIBRARY_API Tensor
{
public:
	Tensor();
	Tensor(size_t size, float value = 0.0f);
	Tensor(const Tensor& other);
	Tensor* operator=(const Tensor& other);
	Tensor(Tensor&& other) noexcept;
	virtual ~Tensor();

	inline const float* GetData() const { return m_Data; }
	float GetAt(size_t i) const;
	void SetAt(size_t i, float value);

	virtual size_t GetSize() const = 0;

	void Map(std::function<float(float v)> mapper) const;
	virtual std::string ToString() const;

protected:
	void Alloc(size_t size);

private:
	float* m_Data = nullptr;
};

namespace_end