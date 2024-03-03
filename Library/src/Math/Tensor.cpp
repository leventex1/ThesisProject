#include "Tensor.h"
#include <assert.h>
#include <sstream>

#include <vector>
#include <future>

#include "../ThreadPool.h"


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

	for (int i = 0; i < size; i++)
	{
		m_Data[i] = value;
	}
}

Tensor::Tensor(size_t size, std::function<float()> initializer)
	: m_Data(nullptr)
{
	Alloc(size);
	for (int i = 0; i < size; i++)
	{
		m_Data[i] = initializer();
	}
}

Tensor::Tensor(const Tensor& other)
	: m_Data(nullptr), m_IsWatcher(false)
{
	size_t size = other.GetSize();
	Alloc(size);

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int dataPerThread = (size + numThreads - 1) / numThreads;

	std::vector<std::future<void>> threads;
	for (size_t i = 0; i < numThreads; i++)
	{
		size_t startRow = i * dataPerThread;
		int endRow = (i == numThreads - 1) ? size : (i + 1) * dataPerThread;

		if (startRow >= size)
			break;

		threads.emplace_back(
			pool->enqueue([startRow, endRow, this, &other] {
				for (size_t i = startRow; i < endRow; ++i)
				{
					size_t otherTrueIndex = other.TraverseTo(i);
					m_Data[i] = other.GetData()[otherTrueIndex];
				}
			})
		);
	}

	for (auto& thread : threads) {
		thread.get();
	}
#else
	for (int i = 0; i < size; i++)
	{
		size_t otherTrueIndex = other.TraverseTo(i);
		m_Data[i] = other.m_Data[otherTrueIndex];
	}
#endif // ASYNC
}

Tensor* Tensor::operator=(const Tensor& other)
{
	size_t size = other.GetSize();
	Alloc(size);
	for(int i = 0; i < size; i++)
	{
		size_t otherTrueIndex = other.TraverseTo(i);
		m_Data[i] = other.m_Data[otherTrueIndex];
	}

	return this;
}

Tensor::Tensor(Tensor&& other) noexcept
	: m_Data(nullptr), m_IsWatcher(other.m_IsWatcher)
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
	for (int i = 0; i < size; i++)
	{
		m_Data[i] = data[i];
	}
}

void AsyncMap(int indexRange, int offset, Tensor* t, std::function<float(float v)> mapper)
{
	for (size_t i = 0; i < indexRange; i++)
	{
		size_t index = offset + i;
		if (index >= t->GetSize())
			break;
		size_t trueIndex = t->TraverseTo(index);
		t->GetData()[trueIndex] = mapper(t->GetData()[trueIndex]);
	}
}

void Tensor::Map(std::function<float(float v)> mapper)
{

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int indexRange = (GetSize() + numThreads - 1) / numThreads;
	std::vector<std::future<void>> threads;

	for (size_t i = 0; i < numThreads; i++)
	{
		//threads.push_back(std::async(std::launch::async, AsyncMap, indexRange, i * indexRange, this, mapper));
		size_t offset = i * indexRange;
		threads.emplace_back(
			pool->enqueue([offset, indexRange, this, mapper] {
				AsyncMap(indexRange, offset, this, mapper);
			})
		);
	}

	for (auto& thread : threads) {
		thread.get();
	}
#else
	for (int i = 0; i < GetSize(); i++)
	{
		size_t trueIndex = TraverseTo(i);
		m_Data[trueIndex] = mapper(m_Data[trueIndex]);
	}
#endif // ASYNC
}

void AsyncElementWise(int indexRange, int offset, Tensor* v1, const Tensor* v2, std::function<float(float v1, float v2)> operation)
{
	for (size_t i = 0; i < indexRange; i++)
	{
		size_t index = offset + i;
		if (index >= v1->GetSize())
			break;
		size_t trueIndex = v1->TraverseTo(index);
		size_t otherTrueIndex = v2->TraverseTo(index);
		v1->GetData()[trueIndex] = operation(v1->GetData()[trueIndex], v2->GetData()[otherTrueIndex]);
	}
}

void Tensor::ElementWise(const Tensor& other, std::function<float(float v1, float v2)> operation)
{
	assert(GetSize() == other.GetSize() && "Tensor sizes not match!");

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int indexRange = (GetSize() + numThreads - 1) / numThreads;
	std::vector<std::future<void>> threads;

	for (size_t i = 0; i < numThreads; i++)
	{
		//threads.push_back(std::async(std::launch::async, AsyncElementWise, indexRange, i * indexRange, this, &other, operation));
		size_t offset = i * indexRange;
		threads.emplace_back(
			pool->enqueue([offset, indexRange, this, &other, operation] {
				AsyncElementWise(indexRange, offset, this, &other, operation);
			})
		);
	}

	for (auto& thread : threads) {
		thread.get();
	}
#else
	for (int i = 0; i < GetSize(); i++)
	{
		size_t trueIndex = TraverseTo(i);
		size_t otherTrueIndex = other.TraverseTo(i);
		m_Data[trueIndex] = operation(m_Data[trueIndex], other.m_Data[otherTrueIndex]);
	}
#endif // ASYNC
}

float Tensor::GetAt(size_t i) const
{
	assert((i < GetSize() || m_IsWatcher) && "Out of index error!");
	return m_Data[i];
}

void Tensor::SetAt(size_t i, float value)
{
	assert((i < GetSize() || m_IsWatcher) && "Out of index error!");
	m_Data[i] = value;
}

std::string Tensor::ToString() const
{
	std::stringstream ss;
	for(size_t i = 0; i < GetSize(); i++)
	{
		size_t trueIndex = TraverseTo(i);
		ss << (i == 0 ? "" : " ") << m_Data[trueIndex];
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
