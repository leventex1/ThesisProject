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
	inline float* GetData() { return m_Data; }
	float GetAt(size_t i) const;
	void SetAt(size_t i, float value);

	virtual size_t GetSize() const = 0;

	void Map(std::function<float(float v)> mapper);
	void ElementWise(const Tensor& other, std::function<float(float v1, float v2)> operation);

	virtual std::string ToString() const;

	inline Tensor& Add(const Tensor& other) { ElementWise(other, [](float v1, float v2) -> float { return v1 + v2; }); return *this; }
	inline Tensor& Sub(const Tensor& other) { ElementWise(other, [](float v1, float v2) -> float { return v1 - v2; }); return *this; }
	inline Tensor& Mult(const Tensor& other) { ElementWise(other, [](float v1, float v2) -> float { return v1 * v2; }); return *this; }
	inline Tensor& Div(const Tensor& other) { ElementWise(other, [](float v1, float v2) -> float { return v1 / v2; }); return *this; }

protected:
	void Alloc(size_t size);

private:
	float* m_Data = nullptr;
};

namespace_end