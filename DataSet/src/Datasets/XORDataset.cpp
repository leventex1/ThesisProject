#include "XORDataset.h"
#include <random>


namespace_dataset_start

static const std::vector<Tensor2D> DataSamples = 
{
		{
			{ 0.0f },
			{ 0.0f }
		},
		{
			{ 1.0f },
			{ 0.0f }
		},
		{
			{ 0.0f },
			{ 1.0f }
		},
		{
			{ 1.0f },
			{ 1.0f }
		}
};

static const std::vector<Tensor2D> DataTargets = 
{
		{
			{ 0.0f },
		},
		{
			{ 1.0f },
		},
		{
			{ 1.0f },
		},
		{
			{ 0.0f },
		}
};

XORDataset::XORDataset()
	: m_SampleIndex(0), m_Samples(DataSamples), m_Labels(DataTargets)
{
}

SampleShape XORDataset::GetSampleShape() const
{
	return {
		2, 1, 1,
		1, 1, 1
	};
}

Sample XORDataset::GetSample() const
{
	return {
		{ m_Samples[m_SampleIndex] },
		{ m_Labels[m_SampleIndex] }
	};
}

void XORDataset::Next()
{
	m_SampleIndex = (m_SampleIndex + 1) % 4;
}

void XORDataset::Shuffle()
{
	std::random_device rd;

	for (size_t i = 0; i < 4; i++)
	{
		size_t swapIndex = rd() % 4;
		std::swap(m_Samples[i], m_Samples[swapIndex]);
		std::swap(m_Labels[i], m_Labels[swapIndex]);
	}

}

namespace_dataset_end