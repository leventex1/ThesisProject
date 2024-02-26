#include "Tensor.h"
#include <assert.h>
#include <sstream>


namespace_start

Tensor::Tensor() : m_Data(nullptr) { }

Tensor::~Tensor()
{
	Dealloc();
}

Tensor::Tensor(size_t size, float value)
	: m_Data(nullptr)
{
	Alloc(size);
	for (size_t i = 0; i < size; i++)
	{
		m_Data[i] = value;
	}
}

Tensor::Tensor(const Tensor& other)
	: m_Data(nullptr)
{
	size_t size = other.GetSize();
	Alloc(size);
	for (size_t i = 0; i < size; i++)
	{
		m_Data[i] = other.m_Data[i];
	}
}

Tensor* Tensor::operator=(const Tensor& other)
{
	size_t size = other.GetSize();
	Alloc(size);
	m_Data = new float[size];
	for(size_t i = 0; i < size; i++)
	{
		m_Data[i] = other.m_Data[i];
	}

	return this;
}

Tensor::Tensor(Tensor&& other) noexcept
	: m_Data(nullptr)
{
	m_Data = other.m_Data;
	other.m_Data = nullptr;
}

Tensor::Tensor(float* m_Watching)
	: m_IsWatcher(true), m_Data((float*)m_Watching)
{
}

Tensor::Tensor(const float* data, size_t size)
{
	Alloc(size);
	for (size_t i = 0; i < size; i++)
	{
		m_Data[i] = data[i];
	}
}

void Tensor::Map(std::function<float(float v)> mapper)
{
	for (size_t i = 0; i < GetSize(); i++)
	{
		m_Data[i] = mapper(m_Data[i]);
	}
}

void Tensor::ElementWise(const Tensor& other, std::function<float(float v1, float v2)> operation)
{
	assert(GetSize() == other.GetSize() && "Tensor sizes not match!");
	for (size_t i = 0; i < GetSize(); i++)
	{
		m_Data[i] = operation(m_Data[i], other.m_Data[i]);
	}
}

float Tensor::GetAt(size_t i) const
{
	assert(i < GetSize() && "Out of index error!");
	return m_Data[i];
}

void Tensor::SetAt(size_t i, float value)
{
	assert(i < GetSize() && "Out of index error!");
	m_Data[i] = value;
}

std::string Tensor::ToString() const
{
	std::stringstream ss;
	for(size_t i = 0; i < GetSize(); i++)
	{
		ss << (i == 0 ? "" : " ") << m_Data[i];
	}
	return ss.str();
}

void Tensor::Alloc(size_t size)
{
	Dealloc();
	m_Data = new float[size];
}

void Tensor::Dealloc()
{
	if (m_Data && !m_IsWatcher)
	{
		delete m_Data;
	}
	m_Data = nullptr;
}

namespace_end
