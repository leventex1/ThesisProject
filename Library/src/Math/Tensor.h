#pragma once
#include "Core.h"
#include <functional>
#include <string>
#include <assert.h>


namespace_start

class LIBRARY_API Tensor
{
public:
	Tensor();
	Tensor(size_t size, float value = 0.0f);
	Tensor(size_t size, std::function<float()> initializer);
	Tensor(const Tensor& other);
	Tensor* operator=(const Tensor& other);
	Tensor(Tensor&& other) noexcept;
	virtual ~Tensor();

	// Ownership is not in the tensor. The memory will not be deallocated when destructed.
	Tensor(float* m_Watching);
	// Copies a block of memory from pointer.
	Tensor(const float* data, size_t size);

	// If the tensor is in watching mode, than use this carefully. Traverse the data with TraverseTo and GetSize.
	inline const float* GetData() const { return m_Data; }
	inline float* GetData() { return m_Data; }
	float GetAt(size_t i) const;
	void SetAt(size_t i, float value);

	virtual size_t GetSize() const = 0;
	// s -> [0, Getsize()), treverse throught the tensor in order. Returns the element's true index in the m_Data array.
	// Use GetData()[TraverseTo(i)], i -> 0, GetSize() to traverse safely the datastructure.
	virtual size_t TraverseTo(size_t s) const { assert("TraverseTo not implemented!"); return s; }; 

	void Map(std::function<float(float v)> mapper);
	void ElementWise(const Tensor& other, std::function<float(float v1, float v2)> operation);

	virtual std::string ToString() const;

	inline Tensor& Add(const Tensor& other) { ElementWise(other, [](float v1, float v2) -> float { return v1 + v2; }); return *this; }
	inline Tensor& Sub(const Tensor& other) { ElementWise(other, [](float v1, float v2) -> float { return v1 - v2; }); return *this; }
	inline Tensor& Mult(const Tensor& other) { ElementWise(other, [](float v1, float v2) -> float { return v1 * v2; }); return *this; }
	inline Tensor& Div(const Tensor& other) { ElementWise(other, [](float v1, float v2) -> float { return v1 / v2; }); return *this; }

protected:
	void Alloc(size_t size);
	void Dealloc();

private:
	bool m_IsWatcher = false;
	float* m_Data = nullptr;
};

namespace_end